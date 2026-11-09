from enum import Enum
from typing import List, Dict, Optional, Any

try:
    from pydantic import BaseModel, Field
except ImportError:
    class _FieldInfo:
        def __init__(self, default=None, default_factory=None, **kwargs):
            self.default = default
            self.default_factory = default_factory

        def get_value(self):
            if self.default_factory is not None:
                return self.default_factory()
            return self.default

    def Field(default=None, default_factory=None, **kwargs):
        return _FieldInfo(default=default, default_factory=default_factory, **kwargs)

    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in self.__class__.__dict__.items():
                if isinstance(v, _FieldInfo):
                    setattr(self, k, v.get_value())
                elif not k.startswith("_") and not callable(v) and not isinstance(v, (property, classmethod)):
                    setattr(self, k, v)
            for k, v in kwargs.items():
                setattr(self, k, v)

        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if hasattr(self, "category") and isinstance(self.category, str):
            try:
                self.category = ScenarioCategory(self.category)
            except ValueError:
                pass


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if hasattr(self, "category") and isinstance(self.category, str):
            try:
                self.category = ScenarioCategory(self.category)
            except ValueError:
                pass

