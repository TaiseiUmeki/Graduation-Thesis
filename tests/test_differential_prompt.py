"""
Phase D (Global) differential prompting tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.attribute_space import ATTR_SPACE, AttributeVector
from models.constraints import Constraint, ConstraintType
from engines.image_generator import ImageGenerator


class _DummyClient:
    pass


def test_prompt_mapping_covers_all_attributes():
    assert set(ImageGenerator.PROMPT_MAPPING.keys()) == set(ATTR_SPACE.all_attributes)


def test_differential_prompt_includes_constrained_attribute():
    gen = ImageGenerator(_DummyClient())

    v0 = AttributeVector(weights={"edge:sharp_angle": 0.8})
    v1 = AttributeVector(weights={"edge:sharp_angle": 0.0})
    constraints = [
        Constraint(
            attribute="edge:sharp_angle",
            constraint_type=ConstraintType.EQUAL,
            value=0.0,
        )
    ]

    prompt = gen._build_image_prompt(
        base_interpretation="base context",
        initial_vector=v0,
        current_vector=v1,
        active_constraints=constraints,
        concept="object",
    )

    assert "rounded edges" in prompt


def test_differential_prompt_includes_changed_attribute():
    gen = ImageGenerator(_DummyClient())

    v0 = AttributeVector(weights={"balance:top_heavy": 0.0})
    v1 = AttributeVector(weights={"balance:top_heavy": 0.85})

    prompt = gen._build_image_prompt(
        base_interpretation="base context",
        initial_vector=v0,
        current_vector=v1,
        active_constraints=[],
        concept="object",
    )

    assert "top-heavy" in prompt

