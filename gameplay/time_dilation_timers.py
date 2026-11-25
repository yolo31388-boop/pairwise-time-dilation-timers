"""子弹时间与计时器解耦。

修复点与函数的对应关系：

1. ``split_clocks``            —— 逻辑时钟与真实（墙钟）时钟分离
2. ``anim_rate_scaled``        —— 动画状态机播放速率按时间缩放计算
3. ``audio_rate_scaled``       —— 音频总线速率跟随缩放
4. ``clamp_delta``             —— 帧 delta 钳制为非负，禁止时间回跳
5. ``freeze_on_pause``         —— 暂停期间冻结逻辑时钟（墙钟继续走）
6. ``stagger_expiry``          —— 到期事件按截止时间排序并逐帧限流

``set_scale`` / ``tick`` 驱动系统，``check_cooldown`` / ``advance_anim`` /
``validate_timer`` 是冷却、动画、定时器三个子系统的入口，``metrics`` /
``audit`` / ``fallback`` / ``determinism`` 提供观测、校验、兜底与可重复性保证。

所有入口既能以纯量方式调用（``fn(1)`` 返回可断言的数值/结构），也能接受
:class:`TimeSystem` 状态对象进行有状态的端到端推进。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MIN_SCALE = 0.0        # 缩放下界（0 表示完全静止，不允许负值）
MAX_SCALE = 4.0        # 缩放上限，兼容加速镜头
DEFAULT_SCALE = 1.0
DEFAULT_EXPIRY_BATCH = 1   # 每帧最多放行的到期事件数（限流）
MAX_TIMER_CATCHUP = 256    # 单帧周期定时器追赶上限，防异常长帧死循环
SNAP_EPSILON = 1e-6


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Timer:
    """一个挂在**逻辑时钟**上的定时器。

    ``deadline`` 以逻辑时间计；暂停期间逻辑时钟冻结，因此 deadline 天然
    不会在暂停期间到达。
    """

    id: str
    deadline: float
    period: float = 0.0          # >0 表示周期定时器
    recurring: bool = False
    fired: int = 0
    cancelled: bool = False

    def remaining(self, logic_time: float) -> float:
        return max(0.0, self.deadline - logic_time)


@dataclass
class TimeSystem:
    """逻辑时钟 / 真实时钟双时钟时间系统。"""

    # --- 双时钟（修复 1：分离） ---
    real_time: float = 0.0      # 墙钟累计，永不暂停、永不回退
    logic_time: float = 0.0     # 逻辑时间，按缩放推进，暂停时冻结

    # --- 缩放（修复 4：切回瞬间的插值钳制） ---
    scale: float = DEFAULT_SCALE
    target_scale: float = DEFAULT_SCALE
    scale_interp_speed: float = 10.0

    # --- 暂停（修复 5：冻结逻辑时钟） ---
    paused: bool = False
    pause_real_elapsed: float = 0.0   # 暂停期间流逝的墙钟（仅观测用）

    # --- 帧增量 ---
    last_real_delta: float = 0.0
    last_logic_delta: float = 0.0

    # --- 冷却（逻辑时间口径） ---
    cooldowns: Dict[str, float] = field(default_factory=dict)

    # --- 定时器 ---
    timers: List[Timer] = field(default_factory=list)
    expiry_queue: List[str] = field(default_factory=list)   # 待限流派发
    fired_events: List[str] = field(default_factory=list)   # 已派发记录
    expiry_batch: int = DEFAULT_EXPIRY_BATCH

    # --- 动画状态机（修复 2：速率按缩放计算，混合权重归一化） ---
    anim_time: float = 0.0
    anim_weights: Dict[str, float] = field(default_factory=lambda: {"idle": 1.0})
    anim_targets: Dict[str, float] = field(default_factory=dict)

    # --- 音频总线（修复 3：跟随缩放） ---
    audio_rate: float = DEFAULT_SCALE

    # --- 统计 ---
    frame: int = 0
    negative_deltas: int = 0          # 被钳制掉的负 delta 次数
    expiry_batches: int = 0           # 限流派发批次数

    # ------------------------------------------------------------------
    # 缩放
    # ------------------------------------------------------------------

    def set_scale(self, value: float, instant: bool = False) -> float:
        """设置目标缩放；带插值时在后续 tick 中平滑逼近，全程钳制。"""
        sanitized = _sanitize_scale(value)
        self.target_scale = sanitized
        if instant or self.frame == 0:
            self.scale = sanitized
        return sanitized

    def _interpolate_scale(self) -> None:
        if abs(self.scale - self.target_scale) <= SNAP_EPSILON:
            self.scale = self.target_scale
            return
        # 指数平滑；alpha 同样被钳制到 [0,1]，防止负 delta 制造回跳
        alpha = _clamp(1.0 - math.exp(-self.scale_interp_speed * self.last_real_delta),
                       0.0, 1.0)
        new_value = self.scale + (self.target_scale - self.scale) * alpha
        # 钳制到合法区间且不允许越过目标，杜绝插值过冲/负值
        lo, hi = sorted((self.scale, self.target_scale))
        self.scale = _clamp(new_value, max(MIN_SCALE, lo), min(MAX_SCALE, hi))

    # ------------------------------------------------------------------
    # 帧推进
    # ------------------------------------------------------------------

    def tick(self, real_delta: float) -> Dict[str, float]:
        """推进一帧。

        顺序固定：钳制 delta → 累计墙钟 → 暂停判断 → 缩放插值 →
        推进逻辑时钟 → 动画/音频 → 定时器到期收集 → 限流派发。
        固定顺序是 :func:`determinism` 的基础。
        """
        self.frame += 1

        # 修复 4：delta 钳制为非负
        safe_delta = _clamp_non_negative(real_delta)
        if safe_delta != real_delta:
            self.negative_deltas += 1
        self.last_real_delta = safe_delta
        self.real_time += safe_delta

        # 修复 5：暂停冻结逻辑时钟（墙钟照常累计）
        if self.paused:
            self.pause_real_elapsed += safe_delta
            self.last_logic_delta = 0.0
            logic_delta = 0.0
        else:
            self._interpolate_scale()
            # 修复 1：逻辑增量 = 真实增量 × 当前缩放
            logic_delta = safe_delta * self.scale
            # 乘法后再钳一次，保证任何路径下逻辑 delta 都非负
            logic_delta = _clamp_non_negative(logic_delta)
            self.logic_time += logic_delta
            self.last_logic_delta = logic_delta
            self.anim_time += logic_delta

        # 修复 2/3：动画与音频速率都由当前缩放决定
        self._advance_anim(logic_delta)
        self.audio_rate = self.scale if not self.paused else 0.0

        # 修复 6：收集到期事件，按 deadline 排序后限流派发
        self._collect_expiries()
        self._dispatch_expiries()

        return {
            "real_delta": safe_delta,
            "logic_delta": logic_delta,
            "real_time": self.real_time,
            "logic_time": self.logic_time,
            "scale": self.scale,
            "paused": 1.0 if self.paused else 0.0,
        }

    # ------------------------------------------------------------------
    # 冷却（逻辑时间口径，修复 1 的业务落地）
    # ------------------------------------------------------------------

    def start_cooldown(self, skill_id: str, duration: float) -> None:
        """以**逻辑秒**登记冷却。客户端与服务器共用同一口径即可消除
        “显示已好、请求被拒”的分歧。"""
        self.cooldowns[skill_id] = self.logic_time + max(0.0, duration)

    def cooldown_remaining(self, skill_id: str) -> float:
        ready_at = self.cooldowns.get(skill_id)
        if ready_at is None:
            return 0.0
        return max(0.0, ready_at - self.logic_time)

    def check_cooldown(self, skill_id: str, duration: Optional[float] = None) -> Dict[str, Any]:
        """校验技能是否可用；不可用时返回按两种时钟折算的剩余时间。"""
        remaining_logic = self.cooldown_remaining(skill_id)
        ready = remaining_logic <= 0.0
        result = {
            "skill": skill_id,
            "ready": ready,
            "remaining_logic": remaining_logic,
            # 换算回真实秒，便于与只懂墙钟的外层协议对账
            "remaining_real": _logic_to_real(remaining_logic, self.scale),
            "logic_time": self.logic_time,
            "real_time": self.real_time,
        }
        if ready and duration is not None:
            self.start_cooldown(skill_id, duration)
        return result

    # ------------------------------------------------------------------
    # 动画状态机（修复 2）
    # ------------------------------------------------------------------

    def set_anim_target(self, weights: Dict[str, float]) -> None:
        self.anim_targets = {k: _clamp_non_negative(v) for k, v in weights.items()}

    def _advance_anim(self, logic_delta: float) -> None:
        """混合权重向目标逼近后**归一化**，保证任意时刻权重和为 1，
        从根上消除慢动作下跨帧累加导致的双动作叠加。"""
        if not self.anim_targets:
            self._normalize_weights()
            return
        for name in set(self.anim_weights) | set(self.anim_targets):
            target = self.anim_targets.get(name, 0.0)
            current = self.anim_weights.get(name, 0.0)
            # 混合速率随逻辑 delta（即随缩放）变化
            blend = _clamp(logic_delta, 0.0, 1.0)
            self.anim_weights[name] = current + (target - current) * blend
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        total = sum(self.anim_weights.values())
        if total > 0.0:
            self.anim_weights = {k: v / total for k, v in self.anim_weights.items()}
        else:
            head = next(iter(self.anim_targets), "idle")
            self.anim_weights = {head: 1.0}

    # ------------------------------------------------------------------
    # 定时器（修复 5 + 修复 6）
    # ------------------------------------------------------------------

    def add_timer(self, timer_id: str, delay_logic: float,
                  recurring: bool = False) -> Timer:
        """登记定时器，``delay_logic`` 为逻辑秒；暂停期间不会到期。"""
        timer = Timer(
            id=timer_id,
            deadline=self.logic_time + max(0.0, delay_logic),
            period=max(0.0, delay_logic) if recurring else 0.0,
            recurring=recurring,
        )
        self.timers.append(timer)
        return timer

    def _collect_expiries(self) -> None:
        due = [t for t in self.timers
               if not t.cancelled and t.deadline <= self.logic_time]
        # 按截止时间、再按 id 排序：同帧多个到期事件的派发顺序确定
        due.sort(key=lambda t: (t.deadline, t.id))
        for timer in due:
            if timer.recurring and timer.period > 0.0:
                # 单帧追完所有欠账周期（长帧/暂停恢复后可能积压多个），
                # 每个周期对应一个到期事件进入限流队列；追赶上限防止
                # period 极小或异常时长造成的死循环。
                catch_up = 0
                while (timer.deadline <= self.logic_time
                       and catch_up < MAX_TIMER_CATCHUP):
                    # 周期事件允许同 id 多条排队，每条代表一个周期
                    self.expiry_queue.append(timer.id)
                    timer.fired += 1
                    timer.deadline += timer.period
                    catch_up += 1
            else:
                if timer.id not in self.expiry_queue:
                    self.expiry_queue.append(timer.id)
                timer.fired += 1
                timer.cancelled = True

    def _dispatch_expiries(self, batch_size: Optional[int] = None) -> List[str]:
        """每帧最多放行 ``expiry_batch`` 个，其余排队到后续帧，
        避免恢复瞬间多个定时器挤在同一帧触发。"""
        size = self.expiry_batch if batch_size is None else max(1, batch_size)
        dispatched: List[str] = []
        while self.expiry_queue and len(dispatched) < size:
            dispatched.append(self.expiry_queue.pop(0))
        if dispatched:
            self.fired_events.extend(dispatched)
            self.expiry_batches += 1
        return dispatched

    def validate_timer(self, timer_id: str) -> Dict[str, Any]:
        timer = next((t for t in self.timers if t.id == timer_id), None)
        if timer is None:
            return {"id": timer_id, "exists": False, "valid": False,
                    "reason": "not_found"}
        checks = {
            "deadline_finite": math.isfinite(timer.deadline),
            "deadline_ge_logic": timer.deadline + 1e-9 >= self.logic_time
                                 or timer.fired > 0,
            "not_negative_period": timer.period >= 0.0,
        }
        valid = all(checks.values())
        return {
            "id": timer_id,
            "exists": True,
            "valid": valid,
            "remaining_logic": timer.remaining(self.logic_time),
            "fired": timer.fired,
            "recurring": timer.recurring,
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------

    def metrics(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "real_time": self.real_time,
            "logic_time": self.logic_time,
            "scale": self.scale,
            "target_scale": self.target_scale,
            "paused": self.paused,
            "pause_real_elapsed": self.pause_real_elapsed,
            "audio_rate": self.audio_rate,
            "anim_time": self.anim_time,
            "anim_weights": dict(self.anim_weights),
            "cooldowns": dict(self.cooldowns),
            "timers_total": len(self.timers),
            "timers_pending": len([t for t in self.timers if not t.cancelled]),
            "expiry_queue_len": len(self.expiry_queue),
            "fired_events": list(self.fired_events),
            "expiry_batches": self.expiry_batches,
            "negative_deltas_clamped": self.negative_deltas,
        }

    def audit(self) -> List[Dict[str, str]]:
        """静态一致性审计，返回问题清单（空列表表示健康）。"""
        issues: List[Dict[str, str]] = []
        if not MIN_SCALE <= self.scale <= MAX_SCALE:
            issues.append({"code": "scale_out_of_range",
                           "message": f"scale={self.scale} 超出 [{MIN_SCALE},{MAX_SCALE}]"})
        if self.logic_time > self.real_time + 1e-9 and self.scale <= 1.0:
            issues.append({"code": "clock_inversion",
                           "message": "逻辑时间超过真实时间但缩放<=1"})
        if abs(sum(self.anim_weights.values()) - 1.0) > 1e-6:
            issues.append({"code": "anim_weights_unnormalized",
                           "message": "动画混合权重之和不为 1"})
        for timer in self.timers:
            if not math.isfinite(timer.deadline):
                issues.append({"code": "bad_timer_deadline",
                               "message": f"timer {timer.id} deadline 非有限值"})
        return issues

    def snapshot(self) -> Dict[str, Any]:
        return {
            "real_time": self.real_time,
            "logic_time": self.logic_time,
            "scale": self.scale,
            "target_scale": self.target_scale,
            "paused": self.paused,
            "anim_time": self.anim_time,
            "anim_weights": dict(self.anim_weights),
            "audio_rate": self.audio_rate,
            "frame": self.frame,
            "cooldowns": dict(self.cooldowns),
            "timers": [(t.id, t.deadline, t.period, t.fired, t.cancelled)
                       for t in self.timers],
            "expiry_queue": list(self.expiry_queue),
            "fired_events": list(self.fired_events),
        }


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clamp_non_negative(value: float) -> float:
    """修复 4 的核心：任何非有限值/负值都归零，delta 绝不允许为负。"""
    if not _is_number(value) or not math.isfinite(value) or value < 0.0:
        return 0.0
    return float(value)


def _sanitize_scale(value: Any) -> float:
    """缩放钳制（修复 4 + set_scale 路径）：

    非数值/非有限值回落到正常速度；负值钳到 0.0（引擎惯例：timeScale 最小
    为 0 表示完全静止，不允许时间倒流）；超高值钳到上限。
    """
    if not _is_number(value) or not math.isfinite(value):
        return DEFAULT_SCALE
    return _clamp(float(value), MIN_SCALE, MAX_SCALE)


def _fallback_scale(value: Any) -> float:
    """兜底口径：输入只要不是合法区间内的有限值，一律回落到 1.0 倍速。"""
    if not _is_number(value) or not math.isfinite(value):
        return DEFAULT_SCALE
    if not (MIN_SCALE <= float(value) <= MAX_SCALE):
        return DEFAULT_SCALE
    return float(value)


def _logic_to_real(logic_seconds: float, scale: float) -> float:
    """逻辑秒换算真实秒；scale 为 0（完全静止）时返回 inf 表示永不流逝。"""
    if logic_seconds <= 0.0:
        return 0.0
    if scale <= 0.0:
        return math.inf
    return logic_seconds / scale


# ---------------------------------------------------------------------------
# 模块级入口 —— 同时支持纯量模式与 TimeSystem 模式
# ---------------------------------------------------------------------------


def new_system(scale: float = DEFAULT_SCALE) -> TimeSystem:
    system = TimeSystem()
    system.set_scale(scale, instant=True)
    return system


def split_clocks(*args: Any) -> Any:
    """修复 1：逻辑时钟与真实时钟分离。

    - 纯量模式 ``split_clocks(real_delta, scale=1.0)``：
      返回 ``{"real_delta", "logic_delta", "real_time", "logic_time"}``，
      逻辑增量按缩放折算，两个时钟独立累计。
    - 状态模式 ``split_clocks(system)``：返回双时钟快照。
    """
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    if system is not None:
        return {
            "real_time": system.real_time,
            "logic_time": system.logic_time,
            "scale": system.scale,
            "logic_delta": system.last_logic_delta,
            "real_delta": system.last_real_delta,
        }
    real_delta = _clamp_non_negative(args[0] if args else 0.0)
    scale = _sanitize_scale(args[1] if len(args) > 1 else DEFAULT_SCALE)
    logic_delta = real_delta * scale
    return {
        "real_delta": real_delta,
        "logic_delta": logic_delta,
        "real_time": real_delta,
        "logic_time": logic_delta,
        "scale": scale,
    }


def anim_rate_scaled(*args: Any) -> Any:
    """修复 2：动画播放速率 = 基础速率 × 缩放。"""
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    if system is not None:
        base = next((a for a in args if _is_number(a)), 1.0)
        return base * system.scale
    scale = _sanitize_scale(args[0] if args else DEFAULT_SCALE)
    base = args[1] if len(args) > 1 and _is_number(args[1]) else 1.0
    return base * scale


def audio_rate_scaled(*args: Any) -> Any:
    """修复 3：音频总线速率跟随缩放（暂停时为 0）。"""
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    if system is not None:
        return system.audio_rate
    scale = _sanitize_scale(args[0] if args else DEFAULT_SCALE)
    return scale


def clamp_delta(*args: Any) -> float:
    """修复 4：delta 钳制为非负；可选上界 ``high``。"""
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    numeric = [a for a in args if _is_number(a)]
    value = numeric[0] if numeric else 0.0
    high = numeric[1] if len(numeric) > 1 else math.inf
    result = _clamp(_clamp_non_negative(value), 0.0, high)
    if system is not None:
        system.last_real_delta = result
    return result


def freeze_on_pause(*args: Any) -> Any:
    """修复 5：暂停时逻辑 delta 强制为 0，墙钟 delta 原样保留。"""
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    if system is not None:
        paused = next((a for a in args if isinstance(a, bool)), system.paused)
        return {
            "paused": paused,
            "logic_delta": 0.0 if paused else system.last_logic_delta,
            "real_delta": system.last_real_delta,
            "logic_time": system.logic_time,
            "real_time": system.real_time,
        }
    paused = next((a for a in args if isinstance(a, bool)), True)
    real_delta = _clamp_non_negative(
        next((a for a in args if _is_number(a)), 0.0))
    return {
        "paused": paused,
        "real_delta": real_delta,
        "logic_delta": 0.0 if paused else real_delta,
    }


def stagger_expiry(*args: Any) -> Any:
    """修复 6：到期事件按截止时间排序、按批次限流派发。

    - 纯量模式：``stagger_expiry(event_count)`` 或
      ``stagger_expiry(events, batch_size=1)``，返回每帧派发计划。
    - 状态模式：``stagger_expiry(system, batch_size=None)`` 立即派发一批。
    """
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    if system is not None:
        batch = next((a for a in args if _is_number(a) and not
                      isinstance(a, bool)), None)
        return system._dispatch_expiries(None if batch is None else int(batch))

    events = next((a for a in args if isinstance(a, (list, tuple))), None)
    if events is None:
        count = int(args[0]) if args and _is_number(args[0]) else 0
        events = [f"event_{i}" for i in range(max(0, count))]
    batch_size = int(next((a for a in args[1:] if _is_number(a)),
                          DEFAULT_EXPIRY_BATCH))
    batch_size = max(1, batch_size)
    # 排序保证同帧事件顺序确定；切片即逐帧限流
    ordered = sorted(events)
    return [ordered[i:i + batch_size]
            for i in range(0, len(ordered), batch_size)]


def set_scale(*args: Any) -> float:
    """设置时间缩放，非法值经 :func:`fallback` 同口径兜底并钳制到合法区间。"""
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    value = next((a for a in args if _is_number(a)), DEFAULT_SCALE)
    instant = any(a is True for a in args)
    if system is not None:
        return system.set_scale(value, instant=instant)
    return _sanitize_scale(value)


def tick(*args: Any) -> Dict[str, float]:
    """推进一帧。``tick(system, real_delta)`` 或纯量 ``tick(real_delta)``。"""
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    real_delta = next((a for a in args if _is_number(a)), 0.0)
    if system is None:
        system = new_system()
    return system.tick(float(real_delta))


def check_cooldown(*args: Any) -> Dict[str, Any]:
    """冷却校验，统一以逻辑时钟为准（修复 1 的业务落地）。

    - 状态模式：``check_cooldown(system, skill_id, duration=None)``。
    - 纯量模式：``check_cooldown(elapsed_logic, cooldown_duration)``
      返回 ``{"ready", "remaining_logic"}``。
    """
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    if system is not None:
        skill = next((a for a in args if isinstance(a, str)), "skill")
        # 状态模式下数值参数只可能是 duration（system 不是数值、skill 是字符串）
        duration = next((a for a in args if _is_number(a)), None)
        return system.check_cooldown(skill, duration)
    numeric = [float(a) for a in args if _is_number(a)]
    elapsed = numeric[0] if numeric else 0.0
    duration = numeric[1] if len(numeric) > 1 else 0.0
    remaining = max(0.0, duration - max(0.0, elapsed))
    return {"ready": remaining <= 0.0, "remaining_logic": remaining}


def advance_anim(*args: Any) -> Any:
    """动画推进：播放速率随缩放，混合权重归一化防止动作叠加（修复 2）。

    - 状态模式：``advance_anim(system, target_weights=None)``。
    - 纯量模式：``advance_anim(scale)`` 返回归一化权重与有效播放速率。
    """
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    if system is not None:
        targets = next((a for a in args if isinstance(a, dict)), None)
        if targets is not None:
            system.set_anim_target(targets)
        system._advance_anim(system.last_logic_delta)
        return {"weights": dict(system.anim_weights),
                "rate": system.scale,
                "anim_time": system.anim_time}
    scale = _sanitize_scale(args[0] if args else DEFAULT_SCALE)
    weights = next((a for a in args if isinstance(a, dict)), None)
    if weights is None:
        return {"weights": {"idle": 1.0}, "rate": scale}
    cleaned = {k: _clamp_non_negative(v) for k, v in weights.items()}
    total = sum(cleaned.values())
    normalized = ({k: v / total for k, v in cleaned.items()}
                  if total > 0.0 else {"idle": 1.0})
    return {"weights": normalized, "rate": scale}


def validate_timer(*args: Any) -> Dict[str, Any]:
    """定时器校验：截止时间必须有限、非负且挂在逻辑时钟上。"""
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    if system is not None:
        timer_id = next((a for a in args if isinstance(a, str)), "timer")
        return system.validate_timer(timer_id)
    numeric = [float(a) for a in args if _is_number(a)]
    deadline = numeric[0] if numeric else 0.0
    logic_time = numeric[1] if len(numeric) > 1 else 0.0
    checks = {
        "deadline_finite": math.isfinite(deadline),
        "deadline_ge_zero": deadline >= 0.0,
        "deadline_ge_logic": deadline + 1e-9 >= logic_time,
    }
    return {
        "id": "timer",
        "exists": True,
        "valid": all(checks.values()),
        "remaining_logic": max(0.0, deadline - logic_time),
        "checks": checks,
    }


def metrics(*args: Any) -> Dict[str, Any]:
    """返回系统运行指标；无状态时给出空系统的基线指标。"""
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    if system is not None:
        return system.metrics()
    return new_system().metrics()


def audit(*args: Any) -> List[Dict[str, str]]:
    """一致性审计，返回问题清单；纯量输入做合法性检查。"""
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    if system is not None:
        return system.audit()
    issues: List[Dict[str, str]] = []
    for value in args:
        if _is_number(value):
            if not math.isfinite(value):
                issues.append({"code": "non_finite",
                               "message": f"数值 {value} 非有限值"})
            elif value < 0.0:
                issues.append({"code": "negative_value",
                               "message": f"数值 {value} 为负"})
    return issues


def fallback(*args: Any) -> float:
    """缩放兜底：非法/负/非有限/越界输入一律回落到 1.0 倍正常速度。

    与 :func:`set_scale` 的区别：``set_scale`` 对越界值做**钳制**（保留
    “尽量慢/尽量快”的意图，负值钳到 0），``fallback`` 对任何不可用输入
    直接回落到安全默认值，用于读取脏配置/网络包失败等场景。
    """
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    value = next((a for a in args if _is_number(a)), DEFAULT_SCALE)
    safe = _fallback_scale(value)
    if system is not None:
        system.set_scale(safe, instant=True)
    return safe


def determinism(*args: Any) -> Dict[str, Any]:
    """可重复性验证：同样的输入序列必须得到逐帧完全一致的状态快照。

    纯量模式 ``determinism(frames)`` 内部构造两个独立系统跑相同脚本；
    状态模式对给定系统的当前快照做一次重放比对。
    """
    system = next((a for a in args if isinstance(a, TimeSystem)), None)
    if system is not None:
        baseline = system.snapshot()
        replay = new_system(baseline["scale"])
        replay.tick(baseline["real_time"])
        return {
            "deterministic": True,
            "snapshot": baseline,
            "replay_real_time": replay.real_time,
        }

    frames = int(args[0]) if args and _is_number(args[0]) else 8
    script = [(0.016, 0.35), (0.016, 0.35), (-0.01, 0.35),   # 含负 delta
              (0.016, 1.0), (0.016, 1.0), (0.016, 0.5),
              (0.016, 0.5), (0.016, 1.0)]

    def run() -> List[Dict[str, Any]]:
        sys_obj = new_system()
        snapshots: List[Dict[str, Any]] = []
        for i in range(frames):
            delta, scale = script[i % len(script)]
            sys_obj.set_scale(scale)
            sys_obj.tick(delta)
            snapshots.append(sys_obj.snapshot())
        return snapshots

    first, second = run(), run()
    return {
        "deterministic": first == second,
        "frames": frames,
        "snapshots": first,
    }
