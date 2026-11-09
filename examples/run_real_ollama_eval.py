import os
import sys
import json
import urllib.request
import urllib.error
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_benchmark.schema import ModelResponse
from eval_benchmark.runner import BenchmarkRunner
from eval_benchmark.report_generator import generate_markdown_report


def query_ollama(model_name: str, prompt: str, timeout: int = 120) -> str:
    """Query local Ollama REST API on http://localhost:11434."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 350
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "").strip()
    except Exception as e:
        print(f"Error querying {model_name}: {e}")
        return f"[ERROR: Failed to query {model_name}]"


def run_real_world_ollama_eval():
    print("=" * 75)
    print("   AI SYSTEMS EVALUATION: LIVE BENCHMARK AGAINST LOCAL FRONTIER LLMS")
    print("=" * 75)

    runner = BenchmarkRunner()
    scenarios = runner.list_scenarios()

    target_models = ["qwen2.5-coder:7b", "phi3:mini"]
    collected_responses = []

    print(f"\nEvaluating {len(target_models)} real models across {len(scenarios)} systems scenarios:")
    for m in target_models:
        print(f"• Model: {m}")

    for model_name in target_models:
        print(f"\n--- Testing Real Model: {model_name} ---")
        for sc in scenarios:
            print(f"Querying [{sc.id}] ({sc.title[:35]}...)...", end="", flush=True)
            t0 = time.perf_counter()
            resp_text = query_ollama(model_name, sc.prompt)
            latency = time.perf_counter() - t0
            print(f" Done ({latency:.1f}s)")

            collected_responses.append(ModelResponse(
                model_name=model_name,
                scenario_id=sc.id,
                response_text=resp_text
            ))

    print(f"\nGrading all {len(collected_responses)} live model responses with 4-Tier Rubric Scorer...")
    results = runner.evaluate_batch(collected_responses)
    summary = runner.aggregate_results(results)

    report_md = generate_markdown_report(results, summary)

    os.makedirs("eval_reports", exist_ok=True)
    report_path = os.path.join("eval_reports", "real_ollama_benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n✅ Official Live Benchmark Report saved to: {report_path}")
    print("\n" + "=" * 75)
    print("                 LIVE MODEL EVALUATION LEADERBOARD")
    print("=" * 75)
    for model, stats in summary.items():
        print(f"• Model: {model:<20} | Score: {stats['average_composite_score']:.2f}/5.0 ({stats['average_percentage']:.1f}%) | Passed: {stats['passed_scenarios']}/{stats['total_scenarios']} ({stats['pass_rate_pct']:.1f}%)")


if __name__ == "__main__":
    run_real_world_ollama_eval()
