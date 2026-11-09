import pytest
from eval_benchmark.schema import ModelResponse
from eval_benchmark.runner import BenchmarkRunner


def test_scorer_catches_cuda_uncoalesced_trap():
    runner = BenchmarkRunner()
    sc = runner.get_scenario("cuda_fused_softmax_coalescing")
    assert sc is not None

    # Flawed response lacking coalesced memory reasoning
    bad_resp = ModelResponse(
        model_name="BadModel",
        scenario_id="cuda_fused_softmax_coalescing",
        response_text="1 thread per row is the most optimal pattern because it avoids thread locks."
    )

    res = runner.evaluate_response(bad_resp)
    assert not res.passed
    assert res.score.hardware_efficiency < 3.0
    assert any("uncoalesced" in t.lower() for t in res.detected_traps)


def test_scorer_rewards_senior_gqa_reasoning():
    runner = BenchmarkRunner()
    good_resp = ModelResponse(
        model_name="SeniorCandidate",
        scenario_id="vllm_llama3_70b_gqa_sizing",
        response_text=(
            "Llama-3-70B uses Grouped-Query Attention (GQA) with 8 KV heads and head dim 128. "
            "Total KV cache for B=64 and S=8192 is 171.8 GB. "
            "Sharded across TP=8, each GPU holds exactly 1 KV head, requiring 20.0 GiB (21.47 GB). "
            "Total allocation per GPU is ~39 GB, fitting within the 80GB H100 without an OOM."
        )
    )

    res = runner.evaluate_response(good_resp)
    assert res.passed
    assert res.score.mathematical_correctness >= 4.5
    assert len(res.detected_traps) == 0

