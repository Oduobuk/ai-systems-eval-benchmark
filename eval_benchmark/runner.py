import json
import os
from typing import List, Dict, Any, Optional
from .schema import Scenario, ModelResponse, EvaluationResult
from .rubric_scorer import RubricScorer


class BenchmarkRunner:
    """
    Executes benchmark evaluations across sets of model responses and scenarios.
    """

    def __init__(self, scenarios_path: Optional[str] = None):
        if scenarios_path is None:
            # Default to bundled benchmark_data/scenarios.json
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            scenarios_path = os.path.join(base_dir, "benchmark_data", "scenarios.json")

        self.scenarios_path = scenarios_path
        self.scenarios: Dict[str, Scenario] = {}
        self.scorer = RubricScorer()
        self._load_scenarios()

    def _load_scenarios(self):
        with open(self.scenarios_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                sc = Scenario(**item)
                self.scenarios[sc.id] = sc

    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        return self.scenarios.get(scenario_id)

    def list_scenarios(self) -> List[Scenario]:
        return list(self.scenarios.values())

    def evaluate_response(self, response: ModelResponse) -> EvaluationResult:
        scenario = self.get_scenario(response.scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario ID '{response.scenario_id}' not found in benchmark suite.")
        return self.scorer.evaluate(scenario, response)

    def evaluate_batch(self, responses: List[ModelResponse]) -> List[EvaluationResult]:
        return [self.evaluate_response(r) for r in responses]

    def aggregate_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Aggregate evaluation metrics grouped by model and category."""
        models: Dict[str, List[EvaluationResult]] = {}
        for res in results:
            models.setdefault(res.model_name, []).append(res)

        summary = {}
        for model_name, model_results in models.items():
            total = len(model_results)
            passed = sum(1 for r in model_results if r.passed)
            avg_score = sum(r.score.composite_score for r in model_results) / total
            avg_pct = sum(r.score.percentage for r in model_results) / total

            # Category scores
            cat_scores: Dict[str, List[float]] = {}
            for r in model_results:
                cat_name = r.category.value if hasattr(r.category, "value") else str(r.category)
                cat_scores.setdefault(cat_name, []).append(r.score.percentage)

            cat_avg = {k: sum(v)/len(v) for k, v in cat_scores.items()}

            summary[model_name] = {
                "total_scenarios": total,
                "passed_scenarios": passed,
                "pass_rate_pct": (passed / total) * 100.0,
                "average_composite_score": avg_score,
                "average_percentage": avg_pct,
                "category_breakdown": cat_avg
            }

        return summary

