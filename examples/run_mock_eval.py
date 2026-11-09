import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_benchmark.schema import ModelResponse
from eval_benchmark.runner import BenchmarkRunner
from eval_benchmark.report_generator import generate_markdown_report


def run_demonstration_evaluation():
    print("=" * 75)
    print("      AI SYSTEMS EVALUATION BENCHMARK: DEMO EVALUATION SUITE")
    print("=" * 75)

    runner = BenchmarkRunner()
    scenarios = runner.list_scenarios()
    print(f"Loaded {len(scenarios)} core systems engineering scenarios.")

    # Candidate A: Senior AI Systems Reference Responses (Clean, Hardware-Aware)
    # Candidate B: Flawed AI Model Responses (Classic Hardware Traps & Hallucinations)
    mock_responses = [
        # Scenario 1: CUDA Softmax
        ModelResponse(
            model_name="Senior-Systems-Candidate",
            scenario_id="cuda_fused_softmax_coalescing",
            response_text=(
                "Mapping 1 thread per row causes severe uncoalesced global memory access across the warp. "
                "Adjacent threads access addresses separated by stride N=1024 floats (4KB apart), requiring 32 separate 128-byte DRAM transactions "
                "and cutting effective memory bus efficiency to 3.1% (4/128 bytes). "
                "To optimize, map 1 block (or 1 warp) per row so threadIdx.x accesses contiguous elements in memory. "
                "Cache row slices in shared memory or registers, and perform row reductions using warp shuffle primitives (__shfl_down_sync)."
            )
        ),
        ModelResponse(
            model_name="Flawed-Model-Baseline",
            scenario_id="cuda_fused_softmax_coalescing",
            response_text=(
                "The proposal is great. By assigning 1 thread per row, we eliminate all thread synchronization overhead and mutexes. "
                "The thread simply iterates through the row, calculates the maximum and sum, and writes the output. "
                "This achieves maximum memory bandwidth because intermediate variables are kept thread-local."
            )
        ),

        # Scenario 2: Llama-3-70B GQA Sizing
        ModelResponse(
            model_name="Senior-Systems-Candidate",
            scenario_id="vllm_llama3_70b_gqa_sizing",
            response_text=(
                "Llama-3-70B uses Grouped-Query Attention (GQA) with 8 KV heads and head dimension d_k = 8192/64 = 128. "
                "KV dimension per token is 8 * 128 = 1024 for both K and V. "
                "Bytes per token across 80 layers = 2 * 80 * 1024 * 2 = 327,680 bytes. "
                "For B=64 and S=8192, total KV cache is 171.8 GB (160 GiB). "
                "Sharding across TP=8 assigns exactly 1 KV head per GPU: 160 GiB / 8 = 20.0 GiB (21.47 GB) per GPU. "
                "Weights take 17.5 GB/GPU. Total allocation per GPU is ~39 GB, fitting easily within 80GB without triggering an OOM."
            )
        ),
        ModelResponse(
            model_name="Flawed-Model-Baseline",
            scenario_id="vllm_llama3_70b_gqa_sizing",
            response_text=(
                "Using the standard attention formula across 64 heads: 2 * 80 * 8192 * 8192 * 64 * 2 = 1.37 TB of KV cache. "
                "Dividing across 8 GPUs yields 171.8 GB per GPU. "
                "Since an H100 only has 80GB of memory, batch size 64 will trigger an immediate OOM error."
            )
        ),

        # Scenario 3: W4A16 Roofline Prefill
        ModelResponse(
            model_name="Senior-Systems-Candidate",
            scenario_id="quantization_w4a16_roofline_prefill",
            response_text=(
                "The claim is false based on the Roofline Model. The Prefill phase consists of large-batch GEMMs operating in the compute-bound regime. "
                "NVIDIA Tensor Cores do not natively execute mixed INT4/FP16 arithmetic; the kernel must unpack and dequantize 4-bit weights into FP16 registers "
                "before computation, adding ALU overhead without increasing peak Tensor Core compute throughput. "
                "W4A16 only accelerates memory-bandwidth bound decode iterations (GEMV), not prefill."
            )
        ),
        ModelResponse(
            model_name="Flawed-Model-Baseline",
            scenario_id="quantization_w4a16_roofline_prefill",
            response_text=(
                "The engineer's claim is completely correct. Compressing weights from 16-bit to 4-bit cuts memory footprint by 4x, "
                "which allows Tensor Cores to run at double the throughput during prefill and significantly speeds up matrix multiplication."
            )
        ),

        # Scenario 4: Optimizer Memory
        ModelResponse(
            model_name="Senior-Systems-Candidate",
            scenario_id="optimizer_adamw_16byte_footprint",
            response_text=(
                "The diagnosis is false. In mixed-precision training, AdamW requires 16 bytes per parameter: "
                "2B weights + 2B gradients + 4B FP32 master weights + 4B 1st momentum + 4B 2nd variance. "
                "For an 8B model, static state alone requires 8e9 * 16B = 128 GB, causing the immediate OOM on a 40GB GPU. "
                "To fix: use 8-bit Adam (bitsandbytes) to compress momentum to 2-4 bytes/param, apply LoRA to train <1% of parameters, or use ZeRO/FSDP."
            )
        ),
        ModelResponse(
            model_name="Flawed-Model-Baseline",
            scenario_id="optimizer_adamw_16byte_footprint",
            response_text=(
                "The engineer is right. The model weights are 16 GB and gradients are 16 GB, which fits in 40GB. "
                "The OOM must be caused by a memory leak in the PyTorch DataLoader retaining tensors between epochs. "
                "To fix this, write a custom Python loop to compute AdamW updates step-by-step."
            )
        )
    ]

    print(f"Running evaluation over {len(mock_responses)} model test responses...")
    results = runner.evaluate_batch(mock_responses)
    summary = runner.aggregate_results(results)

    report_md = generate_markdown_report(results, summary)

    os.makedirs("eval_reports", exist_ok=True)
    report_path = os.path.join("eval_reports", "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nSuccessfully generated benchmark evaluation report: {report_path}")
    print("\n" + "=" * 75)
    print("                     EVALUATION LEADERBOARD SUMMARY")
    print("=" * 75)
    for model, stats in summary.items():
        print(f"• Model: {model:<25} | Score: {stats['average_composite_score']:.2f}/5.0 ({stats['average_percentage']:.1f}%) | Pass Rate: {stats['pass_rate_pct']:.1f}%")


if __name__ == "__main__":
    run_demonstration_evaluation()

