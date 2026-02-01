"""
Phase D (Local) partial edit option generation tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.image_generator import ImageGenerator


class _DummyClient:
    pass


def test_generate_partial_edit_options_positive_delta():
    gen = ImageGenerator(_DummyClient())
    opts = gen.generate_partial_edit_options(
        target_part_name="legs",
        concept="chair",
        attribute_key="edge:knife_edge",
        delta_value=0.8,
    )
    assert len(opts) == 3
    assert opts[0]["type"] == "Literal"
    assert "The legs is" in opts[0]["prompt"]
    # aggressive should pick >=0.85 line for knife_edge
    assert "knife-edge" in opts[1]["prompt"]


def test_generate_partial_edit_options_negative_delta_uses_antonym():
    gen = ImageGenerator(_DummyClient())
    opts = gen.generate_partial_edit_options(
        target_part_name="handle",
        concept="cup",
        attribute_key="edge:knife_edge",
        delta_value=-0.4,
    )
    assert len(opts) == 3
    # 0.00 mapping for knife_edge is an "avoid thin knife edges..." style
    assert "avoid thin knife edges" in opts[0]["prompt"]

