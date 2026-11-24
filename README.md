# pairwise-time-dilation-timers

## 领域

游戏、图形与多媒体

## 主题

子弹时间与计时器解耦

## 说明

逻辑时钟分离、动画速率与到期限流。

初始状态下 `gameplay/time_dilation_timers.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
