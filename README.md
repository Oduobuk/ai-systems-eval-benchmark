# AI Systems Evaluation Benchmark: Auditing Foundation Models on Systems Engineering

[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)](https://github.com/Oduobuk/ai-systems-eval-benchmark)
[![Evaluation](https://img.shields.io/badge/Task-AI%20Systems%20%7C%20Model%20Evaluation-purple)](#)
[![Domain](https://img.shields.io/badge/Domain-CUDA%20%7C%20vLLM%20%7C%20Quantization-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A standardized evaluation benchmark and automated rubric scoring harness designed to **evaluate, stress-test, and benchmark the technical reasoning of frontier Large Language Models (LLMs)** on low-level systems engineering, GPU kernel optimization, and high-throughput inference serving.

Built from first principles in Python and Pydantic, this framework automates **diagnostic trap detection, 4-tier rubric scoring, canonical reference comparison, and leaderboard report generation**, exposing when models produce syntactically valid code that suffers from critical hardware bottlenecks (such as uncoalesced memory access, GQA dimension inflation, Roofline prefill fallacies, or optimizer memory miscalculations).

---

## 🏛️ The Problem: Why Standard Code Evals Miss Systems Failures

Existing AI benchmarks (e.g., HumanEval, MBPP) primarily evaluate high-level algorithmic logic (arrays, string manipulation, recursion). However, when foundation models are tasked with systems-level engineering:
* **The "Plausible Hallucination":** A model generates pristine C++/CUDA code that compiles cleanly, but strides across memory in a manner that serializes warps and drops memory bus efficiency to 3.1%.
* **The "Hardware Disconnect":** A model claims that 4-bit weight quantization (W4A16) doubles compute throughput during large-batch prompt prefill on Tensor Cores, ignoring that dequantization adds ALU overhead to compute-bound GEMM.
* **The "Architecture Blindspot":** A model sizes the KV cache of a 70B model using naive Multi-Head Attention (MHA) formulas rather than Grouped-Query Attention (GQA), inflating memory by $8\times$ and falsely diagnosing an unavoidable Out-Of-Memory (OOM) error.

Evaluating systems reasoning requires an evaluation harness built on **first-principles hardware constraints and diagnostic error taxonomies**.

---

## ⚙️ The 4-Tier Systems Evaluation Matrix

Each model solution is evaluated across four standardized, orthogonal dimensions on a 0.0 to 5.0 scale:

```
                  THE 4-TIER EVALUATION TAXONOMY
     +-------------------------------------------------------+
     |                 EVALUATION TAXONOMY                   |
     +-------------------------------------------------------+
                                |
     +--------------------------+----------------------------+
     |                                                       |
     v                                                       v
+-------------------------------+       +------------------------------------+
|   1. MATHEMATICAL RIGOR       |       |       2. HARDWARE EFFICIENCY       |
| • Tensor dimensional balance  |       | • Coalesced memory transactions    |
| • GQA head partition (TP=8)   |       | • Roofline compute vs memory bound |
| • Static optimizer accounting |       | • Warp shuffle vs atomic reduction |
+-------------------------------+       +------------------------------------+
                                |
     +--------------------------+----------------------------+
     |                                                       |
     v                                                       v
+-------------------------------+       +------------------------------------+
|    3. ARCHITECTURAL SAFETY    |       |        4. CLARITY & DEPTH          |
| • OOM boundary handling       |       | • Root-cause explanation depth     |
| • Preemption / eviction logic |       | • Avoidance of generic buzzwords   |
| • Zero-trust data isolation   |       | • Canonical reference solutions    |
+-------------------------------+       +------------------------------------+
```

---

## 📁 Repository Structure

```
ai-systems-eval-benchmark/
├── benchmark_data/
│   └── scenarios.json              # 6+ curated expert systems engineering scenarios
├── eval_benchmark/
│   ├── __init__.py                 # Clean package entrypoint
│   ├── schema.py                   # Pydantic data models (Scenario, Response, Score)
│   ├── rubric_scorer.py            # Diagnostic trap detector & 4-tier rubric scorer
│   ├── runner.py                   # Batch evaluation runner & metrics aggregator
│   └── report_generator.py         # Markdown leaderboard & report formatter
├── examples/
│   └── run_mock_eval.py            # End-to-end evaluation demonstration script
├── tests/
│   ├── test_schema.py              # Scenario dataset validation tests
│   └── test_rubric_scorer.py       # Scorer sensitivity & trap detection tests
├── Makefile                        # Top-level test & demo targets
├── pyproject.toml                  # Python package configuration
└── README.md                       # Evaluation documentation & rubrics
```

---

## 🔬 Sample Benchmark Leaderboard

Running the evaluation engine over candidate model responses across the benchmark suite:

| Rank | Model Evaluated | Avg Score | Percentage | Passed / Total | Pass Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **#1** | **Senior-Systems-Candidate** | **4.98 / 5.0** | **99.5%** | **4 / 4** | **100.0%** |
| **#2** | **Flawed-Model-Baseline** | **1.72 / 5.0** | **34.5%** | **0 / 4** | **0.0%** |

### Category Breakdown
| Model | CUDA Kernel | Inference Serving | Quantization | Optimizer Design |
| :--- | :--- | :--- | :--- | :--- |
| **Senior-Systems-Candidate** | **100.0%** | **100.0%** | **98.0%** | **100.0%** |
| **Flawed-Model-Baseline** | 32.0% | 35.0% | 25.0% | 46.0% |

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/Oduobuk/ai-systems-eval-benchmark.git
cd ai-systems-eval-benchmark
pip install -e .
```

### 2. Run Test Suite
```bash
make test
```

### 3. Run Benchmark Demo
```bash
make demo
```

### 4. Python API Usage
```python
from eval_benchmark import BenchmarkRunner, ModelResponse

# 1. Initialize benchmark runner
runner = BenchmarkRunner()

# 2. Evaluate a model response
response = ModelResponse(
    model_name="Llama-3-70B-Instruct",
    scenario_id="cuda_fused_softmax_coalescing",
    response_text="Mapping 1 thread per row causes uncoalesced global memory access..."
)

result = runner.evaluate_response(response)

print(f"Passed: {result.passed}")
print(f"Composite Score: {result.score.composite_score:.2f} / 5.0")
print(result.evaluator_critique)
```

---

## 📄 License
MIT License - see [LICENSE](LICENSE) for details.
