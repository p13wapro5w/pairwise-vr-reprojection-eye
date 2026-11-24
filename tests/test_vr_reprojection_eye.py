import pytest
import graphics.vr_reprojection_eye as mod

CASES = [
    ("pose_predict", (1,)),
    ("vsync_align", (1,)),
    ("eye_history", ("left",)),
    ("shared_depth", (1,)),
    ("twist_cap", (1,)),
    ("drift_correct", ((0.0, 0.0, 0.0),)),
    ("submit_match", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("late_latch", (1,)),
    ("metrics", (1,)),
    ("asymmetric", ("left",)),
    ("compose", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
