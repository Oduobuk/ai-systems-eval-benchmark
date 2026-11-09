import os
import json
import pytest
from eval_benchmark.schema import Scenario, ScenarioCategory


def test_scenarios_json_validity():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scenarios_path = os.path.join(base_dir, "benchmark_data", "scenarios.json")

    assert os.path.exists(scenarios_path), "scenarios.json not found"

    with open(scenarios_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 6, "Expected at least 6 scenarios"

    for item in data:
        sc = Scenario(**item)
        assert sc.id
        assert sc.title
        assert sc.category in ScenarioCategory
        assert len(sc.key_traps) > 0
        assert len(sc.golden_reference) > 50
