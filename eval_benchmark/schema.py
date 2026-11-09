from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class ScenarioCategory(str, Enum):
    CUDA_KERNEL = "CUDA_KERNEL"
    INFERENCE_SERVING = "INFERENCE_SERVING"
    QUANTIZATION = "QUANTIZATION"
    OPTIMIZER_DESIGN = "OPTIMIZER_DESIGN"
    ATTENTION_MECHANICS = "ATTENTION_MECHANICS"


class Scenario(BaseModel):
    id: str
    title: str
    category: ScenarioCategory
    difficulty: str
    prompt: str
    key_traps: List[str]
    golden_reference: str
    rubric: Dict[str, int] = Field(default_factory=lambda: {
        "mathematical_correctness": 5,
        "hardware_efficiency": 5,
        "architectural_safety": 5,
        "depth_of_explanation": 5
    })


class ModelResponse(BaseModel):
    model_name: str
    scenario_id: str
    response_text: str


class RubricScore(BaseModel):
    mathematical_correctness: float = Field(ge=0.0, le=5.0)
    hardware_efficiency: float = Field(ge=0.0, le=5.0)
    architectural_safety: float = Field(ge=0.0, le=5.0)
    depth_of_explanation: float = Field(ge=0.0, le=5.0)

    @property
    def composite_score(self) -> float:
        """Weighted average score out of 5.0 (scaled to 100%)."""
        return (
            self.mathematical_correctness * 0.30 +
            self.hardware_efficiency * 0.30 +
            self.architectural_safety * 0.20 +
            self.depth_of_explanation * 0.20
        )

    @property
    def percentage(self) -> float:
        return (self.composite_score / 5.0) * 100.0


class EvaluationResult(BaseModel):
    scenario_id: str
    scenario_title: str
    category: ScenarioCategory
    model_name: str
    score: RubricScore
    detected_traps: List[str]
    evaluator_critique: str
    passed: bool

