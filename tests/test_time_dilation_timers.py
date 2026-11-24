import pytest
import gameplay.time_dilation_timers as mod

CASES = [
    ("split_clocks", (1,)),
    ("anim_rate_scaled", (1,)),
    ("audio_rate_scaled", (1,)),
    ("clamp_delta", (1,)),
    ("freeze_on_pause", (1,)),
    ("stagger_expiry", (1,)),
    ("set_scale", (1,)),
    ("tick", (1,)),
    ("check_cooldown", (1,)),
    ("advance_anim", (1,)),
    ("validate_timer", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
