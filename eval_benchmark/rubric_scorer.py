import re
from typing import List, Tuple
from .schema import Scenario, ModelResponse, RubricScore, EvaluationResult, ScenarioCategory


class RubricScorer:
    """
    Automated 4-Tier Rubric Evaluation Engine for AI Systems Engineering.
    
    Evaluates solutions across four standardized dimensions:
    1. Mathematical Correctness (0.0 - 5.0)
    2. Hardware & Resource Efficiency (0.0 - 5.0)
    3. Architectural Safety & Robustness (0.0 - 5.0)
    4. Depth of Technical Explanation (0.0 - 5.0)
    """

    def evaluate(self, scenario: Scenario, response: ModelResponse) -> EvaluationResult:
        text = response.response_text.lower()
        traps_detected: List[str] = []

        math_score = 5.0
        hw_score = 5.0
        safety_score = 5.0
        depth_score = 5.0

        # Scenario-specific diagnostic pattern rules
        if scenario.id == "cuda_fused_softmax_coalescing":
            if "uncoalesced" not in text and "coalesc" not in text:
                traps_detected.append("Missed global memory uncoalesced access pattern.")
                hw_score -= 2.5
            if "warp" not in text and "shuffle" not in text and "__shfl" not in text:
                traps_detected.append("Omitted warp shuffle reduction primitive (__shfl_down_sync).")
                hw_score -= 1.5
            if "shared memory" not in text and "__shared__" not in text:
                traps_detected.append("Did not utilize on-chip shared memory / registers to cache row data.")
                hw_score -= 1.0

        elif scenario.id == "vllm_llama3_70b_gqa_sizing":
            # Check for GQA awareness (8 KV heads vs 64 heads)
            if "1.37 tb" in text or "1.37tb" in text or "1374" in text or "64 heads" in text and "8 kv" not in text:
                traps_detected.append("Ignored Grouped-Query Attention (GQA 8 KV heads); used naive MHA formula.")
                math_score -= 3.0
            if "oom" in text and ("trigger an oom" in text or "cause an oom" in text or "will oom" in text):
                traps_detected.append("Falsely claimed B=64 will trigger an Out-Of-Memory error.")
                math_score -= 1.5
            if "20" not in text and "21" not in text and "160" not in text:
                traps_detected.append("Failed to derive accurate per-GPU KV footprint (~20-21.5 GB).")
                math_score -= 1.0

        elif scenario.id == "quantization_w4a16_roofline_prefill":
            if "correct" in text and ("engineer is correct" in text or "claim is correct" in text or "will double" in text):
                traps_detected.append("Incorrectly agreed that W4A16 doubles prefill execution throughput.")
                math_score -= 3.0
                hw_score -= 2.5
            if "compute-bound" not in text and "compute bound" not in text:
                traps_detected.append("Failed to identify that prompt prefill operates in the compute-bound Roofline regime.")
                hw_score -= 1.5
            if "dequant" not in text and "unpack" not in text:
                traps_detected.append("Omitted dynamic dequantization overhead on Tensor Cores.")
                depth_score -= 1.5

        elif scenario.id == "optimizer_adamw_16byte_footprint":
            if "dataloader" in text and ("leak" in text and "adamw" not in text):
                traps_detected.append("Falsely attributed OOM to DataLoader memory leak.")
                safety_score -= 2.0
            if "16 bytes" not in text and "16b" not in text and "128 gb" not in text and "128gb" not in text:
                traps_detected.append("Omitted 16 bytes/param static training footprint breakdown.")
                math_score -= 2.0
            if "master" not in text and "fp32" not in text:
                traps_detected.append("Forgot 4-byte FP32 master weights requirement.")
                depth_score -= 1.5

        elif scenario.id == "sglang_radix_attention_prefix_caching":
            if "radix" not in text and "trie" not in text and "tree" not in text:
                traps_detected.append("Failed to explain Radix Tree hierarchical cache structure.")
                depth_score -= 2.0
            if "lru" not in text:
                traps_detected.append("Omitted LRU cache eviction mechanism.")
                safety_score -= 1.5

        elif scenario.id == "speculative_decoding_rejection_sampling":
            if "residual" not in text and "p - q" not in text and "p_res" not in text:
                traps_detected.append("Failed to define the normalized positive residual distribution.")
                math_score -= 2.5
            if "approximation" in text or "degrades quality" in text:
                traps_detected.append("Falsely claimed speculative decoding is an approximation causing quality loss.")
                safety_score -= 2.0

        # General text depth and clarity assessment
        word_count = len(response.response_text.split())
        if word_count < 60:
            depth_score = min(depth_score, 2.0)
            safety_score = min(safety_score, 2.5)

        # Clamp all scores to [0.0, 5.0]
        math_score = max(0.0, min(5.0, math_score))
        hw_score = max(0.0, min(5.0, hw_score))
        safety_score = max(0.0, min(5.0, safety_score))
        depth_score = max(0.0, min(5.0, depth_score))

        score = RubricScore(
            mathematical_correctness=math_score,
            hardware_efficiency=hw_score,
            architectural_safety=safety_score,
            depth_of_explanation=depth_score
        )

        passed = score.percentage >= 70.0 and len(traps_detected) <= 1

        critique = self._build_critique(scenario, traps_detected, score, passed)

        return EvaluationResult(
            scenario_id=scenario.id,
            scenario_title=scenario.title,
            category=scenario.category,
            model_name=response.model_name,
            score=score,
            detected_traps=traps_detected,
            evaluator_critique=critique,
            passed=passed
        )

    def _build_critique(
        self,
        scenario: Scenario,
        traps: List[str],
        score: RubricScore,
        passed: bool
    ) -> str:
        status_str = "PASS" if passed else "FAIL"
        critique_lines = [
            f"[{status_str}] Overall Score: {score.composite_score:.2f} / 5.0 ({score.percentage:.1f}%)",
            f"• Mathematical Correctness: {score.mathematical_correctness:.1f} / 5.0",
            f"• Hardware Efficiency:      {score.hardware_efficiency:.1f} / 5.0",
            f"• Architectural Safety:     {score.architectural_safety:.1f} / 5.0",
            f"• Depth of Explanation:     {score.depth_of_explanation:.1f} / 5.0",
        ]

        if traps:
            critique_lines.append("\nDetected Reasoning Flaws & Traps:")
            for t in traps:
                critique_lines.append(f"  ❌ {t}")
        else:
            critique_lines.append("\nClean evaluation: No critical hardware or mathematical traps detected.")

        return "\n".join(critique_lines)

