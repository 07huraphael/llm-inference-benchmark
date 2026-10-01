# LLM Inference Benchmark

A systems-oriented project for studying the performance characteristics of local LLM inference on a CPU-based Windows laptop.

This project started as a simple benchmark of LLM output length and gradually developed into an investigation of system-level factors such as CPU temperature, effective clock, and power limits.

---

## Project Goal

The main goal of this project is to understand:

> What factors affect the performance of local LLM inference?

Rather than focusing on model accuracy, this project investigates the **systems side of AI inference**, including:

- Latency
- Tokens per second
- Output length
- Run-to-run variation
- CPU temperature
- Effective CPU clock
- CPU package power
- Power limits
- Thermal throttling

---

## Model

- **Qwen2.5-0.5B-Instruct**

The model is small enough to run locally on CPU while still allowing meaningful inference-performance experiments.

---

## Environment

- Windows
- Python 3.11
- PyTorch
- Hugging Face Transformers
- CPU inference
- Intel Core i5-1240P
- Intel Iris Xe Graphics
- No NVIDIA CUDA GPU

---

# Experiment 1 — Output Length Benchmark

## Research Question

How does output length affect LLM inference latency and generation speed?

## Baseline Inference

A single inference was first performed to measure:

- Generated tokens
- Total latency
- Tokens per second

This established a baseline for later experiments.

## Repeated Benchmark

The same inference was repeated multiple times because system performance can vary between runs.

Possible sources of variation include:

- Background processes
- CPU load
- CPU temperature
- Power-management behavior
- Memory pressure

A warm-up inference was also performed before benchmarking to reduce first-run initialization effects.

## Output Length Benchmark

The model was tested with the following output lengths:

- 16 tokens
- 32 tokens
- 64 tokens
- 128 tokens
- 256 tokens

For each output length, latency and generation speed were measured repeatedly.

## Randomized Benchmark Order

The first experiment always tested output lengths in ascending order.

This introduced a possible bias because longer outputs were always tested later, when the CPU might already have been under sustained load.

The improved benchmark therefore randomized the test order for each round.

Example:

```text
Round 1
64 → 256 → 16 → 128 → 32

Round 2
256 → 32 → 128 → 64 → 16
```

This reduced execution-order bias.

---

## Experiment 1 Results

The experiment showed a clear increase in total latency as output length increased.

However, generation speed did not decrease monotonically with output length.

Approximate average results from the randomized benchmark:

| Output Length | Mean Latency | Mean Throughput |
|---:|---:|---:|
| 16 tokens | 2.36 s | 8.53 tokens/s |
| 32 tokens | 5.36 s | 8.86 tokens/s |
| 64 tokens | 10.79 s | 7.72 tokens/s |
| 128 tokens | 15.26 s | 9.36 tokens/s |
| 256 tokens | 31.12 s | 8.48 tokens/s |

The most interesting observation was not the output length itself, but the large **run-to-run variation**.

Some benchmark rounds were significantly slower than others even under the same model and output settings.

This led to a new question:

> Why does LLM inference performance vary so much between runs?

---

## Experiment 1 Visualizations

### Output Length vs Latency

![Output Length vs Latency](graphs/output_length_vs_latency_v2.png)

### Output Length vs Generation Speed

![Output Length vs Generation Speed](graphs/output_length_vs_speed_v2.png)

---

# Experiment 2 — Thermal and Power Investigation

## Motivation

The large run-to-run variation observed in Experiment 1 suggested that hardware state might be affecting inference performance.

This led to an investigation of:

- CPU temperature
- Effective CPU clock
- CPU package power
- Dynamic PL1 power limit
- Thermal throttling
- LLM throughput

---

## Research Question

Does CPU temperature reduce LLM performance directly, or is performance degradation more closely related to a reduction in effective CPU clock caused by thermal throttling?

---

## Hypothesis

> High CPU temperature causes thermal throttling, which reduces effective CPU clock and therefore decreases LLM inference throughput.

---

## Experimental Method

The same model generated **128 tokens continuously for 20 runs**.

At the same time, HWiNFO recorded hardware sensor data.

The LLM benchmark recorded:

- Run number
- Start time
- End time
- Latency
- Tokens per second

HWiNFO recorded:

- CPU temperature
- Effective CPU clock
- CPU package power
- Dynamic PL1
- Thermal-throttling status
- Power-limit status
- Memory usage

The two datasets were matched using timestamps.

---

## Experiment 2 Results

### LLM Performance

| Metric | Result |
|---|---:|
| Mean throughput | 6.36 tokens/s |
| Minimum throughput | 3.95 tokens/s |
| Maximum throughput | 8.30 tokens/s |
| Mean latency | 21.02 s |
| Minimum latency | 15.41 s |
| Maximum latency | 32.38 s |

### CPU State During Measured Runs

| Metric | Result |
|---|---:|
| Mean CPU temperature | ~70.2 °C |
| Run-average temperature range | ~69.4–71.0 °C |
| Mean effective CPU clock | ~858 MHz |
| Run-average effective-clock range | ~815–892 MHz |
| CPU package power | ~12 W |
| Dynamic PL1 | 12 W |
| Thermal throttling during measured runs | Not observed |
| Power-limit condition | Observed |

The exploratory Pearson correlation between effective CPU clock and LLM throughput was approximately:

**r ≈ +0.82**

Runs with a higher effective CPU clock generally showed higher LLM throughput.

However, correlation alone does not prove causation.

---

## Experiment 2 Interpretation

The original hypothesis predicted:

```text
CPU temperature increase
        ↓
Thermal throttling
        ↓
Effective clock reduction
        ↓
Lower LLM throughput
```

However, this sequence was **not observed during the 20 measured inference runs**.

CPU temperature remained close to 70 °C, and no thermal throttling occurred.

Despite this, throughput varied substantially from approximately:

```text
3.95 tokens/s
to
8.30 tokens/s
```

Therefore, thermal throttling does not appear to explain the measured performance variation.

At the same time:

- Dynamic PL1 remained at approximately 12 W.
- Power-limit conditions were observed.
- Effective CPU clock showed a strong positive association with LLM throughput.

This suggests a new hypothesis:

> Sustained CPU power limits may reduce effective CPU clock and contribute to LLM inference-performance degradation.

---

## Important Observation

The HWiNFO log also recorded an earlier period before the measured LLM runs in which:

- CPU temperature exceeded 90 °C.
- Thermal throttling occurred.
- Dynamic PL1 decreased toward 12 W.

However, because this occurred before the measured benchmark runs, the current experiment cannot establish that thermal throttling caused the later 12 W power limit.

A controlled follow-up experiment is required.

---

## Thermal Experiment Visualizations

### Throughput Across Runs

![Throughput Across Runs](thermal/graphs/run_vs_throughput.png)

### Effective CPU Clock vs LLM Throughput

![Effective Clock vs Throughput](thermal/graphs/clock_vs_throughput.png)

### CPU Temperature vs LLM Throughput

![Temperature vs Throughput](thermal/graphs/temperature_vs_throughput.png)

For the full experiment:

[Read the Thermal and Power Investigation](thermal/README.md)

---

# Current Findings

So far, the project suggests:

1. **Longer outputs clearly increase total inference latency.**
2. **Generation speed does not simply decrease as output length increases.**
3. **Run-to-run performance variation can be substantial.**
4. **CPU temperature alone did not explain the observed performance variation.**
5. **No thermal throttling occurred during the measured thermal benchmark runs.**
6. **Effective CPU clock showed a strong positive association with LLM throughput.**
7. **CPU power limits are a promising factor for further investigation.**

---

# Limitations

The current experiments have several limitations:

- Only one laptop and one CPU were tested.
- All inference experiments were CPU-based.
- Windows background processes were not fully controlled.
- Physical memory usage was high during the thermal experiment.
- CPU power-management behavior was not independently controlled.
- The observed relationships are correlations and do not establish causality.
- The CPU temperature range during the measured thermal runs was relatively narrow.

---

# Project Structure

```text
llm-inference-benchmark/
│
├── src/
│   ├── baseline.py
│   ├── benchmark.py
│   ├── benchmark_lengths.py
│   ├── benchmark_lengths_v2.py
│   ├── plot_results.py
│   ├── plot_results_v2.py
│   ├── thermal_benchmark.py
│   └── analyze_thermal.py
│
├── results/
│   ├── output_length_benchmark.csv
│   └── output_length_benchmark_v2.csv
│
├── graphs/
│   ├── output_length_vs_latency.png
│   ├── output_length_vs_latency_v2.png
│   ├── output_length_vs_speed.png
│   └── output_length_vs_speed_v2.png
│
├── thermal/
│   ├── README.md
│   │
│   ├── results/
│   │   ├── llm_thermal_benchmark.csv
│   │   └── thermal_summary.csv
│   │
│   └── graphs/
│       ├── run_vs_throughput.png
│       ├── clock_vs_throughput.png
│       └── temperature_vs_throughput.png
│
├── .gitignore
└── README.md
```

---

# Next Experiment

The next hypothesis is:

> Sustained CPU power limits (PL1) reduce effective CPU clock, which decreases LLM inference throughput.

The next experiment will attempt to capture the complete transition from a relatively cool CPU state to sustained load while simultaneously measuring:

- CPU temperature
- CPU package power
- Dynamic PL1
- Effective CPU clock
- Thermal throttling
- LLM throughput

Future extensions may also include:

- CPU thread-count experiments
- Memory-pressure experiments
- Windows power-mode comparison
- CPU vs GPU inference
- FP32 vs FP16 inference
- KV-cache analysis
- Per-token latency
- Time to First Token (TTFT)
- vLLM serving benchmarks