import pandas as pd
import matplotlib.pyplot as plt


LLM_PATH = "thermal/results/llm_thermal_benchmark.csv"
HWINFO_PATH = "thermal/results/hwinfo_log.csv"

SUMMARY_PATH = "thermal/results/thermal_summary.csv"


# --------------------------------------------------
# Load LLM benchmark
# --------------------------------------------------

llm = pd.read_csv(LLM_PATH)

llm["start_time"] = pd.to_datetime(llm["start_time"])
llm["end_time"] = pd.to_datetime(llm["end_time"])


# --------------------------------------------------
# Load HWiNFO log
# --------------------------------------------------

hw = pd.read_csv(
    HWINFO_PATH,
    encoding="utf-8-sig"
)

# HWiNFO CSV 마지막 부분에 반복 헤더 등이 들어갈 수 있으므로
# 정상적인 날짜 행만 남긴다.
hw = hw[
    hw["Date"]
    .astype(str)
    .str.match(r"\d+\.\d+\.\d+")
].copy()

hw["timestamp"] = pd.to_datetime(
    hw["Date"] + " " + hw["Time"],
    format="%d.%m.%Y %H:%M:%S.%f"
)


# HWiNFO에는 "CPU 전체" 센서가 두 개 있을 수 있다.
# 현재 로그에서는 .1 항목을 CPU Enhanced temperature로 사용한다.
if "CPU 전체 [°C].1" in hw.columns:
    TEMP_COLUMN = "CPU 전체 [°C].1"
else:
    TEMP_COLUMN = "CPU 전체 [°C]"

CLOCK_COLUMN = "평균 유효 클럭 [MHz]"
POWER_COLUMN = "CPU 패키지 전력 소비 [W]"
PL1_COLUMN = "PL1 전력 제한 (Dynamic) [W]"

THERMAL_COLUMN = "코어 열 조절 (avg) [Yes/No]"
POWER_LIMIT_COLUMN = "코어 전력 제한 초과 (avg) [Yes/No]"

MEMORY_COLUMN = "물리적 메모리 사용량 [%]"


numeric_columns = [
    TEMP_COLUMN,
    CLOCK_COLUMN,
    POWER_COLUMN,
    PL1_COLUMN,
    MEMORY_COLUMN,
]

for column in numeric_columns:
    hw[column] = pd.to_numeric(
        hw[column],
        errors="coerce"
    )


# --------------------------------------------------
# Match hardware samples to each LLM run
# --------------------------------------------------

results = []

for _, run in llm.iterrows():

    segment = hw[
        (hw["timestamp"] >= run["start_time"])
        &
        (hw["timestamp"] <= run["end_time"])
    ]

    results.append({
        "run": run["run"],
        "latency": run["latency"],
        "tokens_per_sec": run["tokens_per_sec"],

        "avg_cpu_temp_c":
            segment[TEMP_COLUMN].mean(),

        "avg_effective_clock_mhz":
            segment[CLOCK_COLUMN].mean(),

        "avg_package_power_w":
            segment[POWER_COLUMN].mean(),

        "dynamic_pl1_w":
            segment[PL1_COLUMN].mean(),

        "thermal_throttling":
            (
                segment[THERMAL_COLUMN]
                .astype(str)
                .eq("예")
                .any()
            ),

        "power_limit_exceeded":
            (
                segment[POWER_LIMIT_COLUMN]
                .astype(str)
                .eq("예")
                .any()
            ),

        "memory_usage_percent":
            segment[MEMORY_COLUMN].mean()
    })


summary = pd.DataFrame(results)

summary.to_csv(
    SUMMARY_PATH,
    index=False
)


# --------------------------------------------------
# Basic analysis
# --------------------------------------------------

clock_correlation = (
    summary["avg_effective_clock_mhz"]
    .corr(summary["tokens_per_sec"])
)

temperature_correlation = (
    summary["avg_cpu_temp_c"]
    .corr(summary["tokens_per_sec"])
)


print("\n--- Thermal Benchmark Summary ---")

print(
    f"Average throughput: "
    f"{summary['tokens_per_sec'].mean():.2f} tokens/sec"
)

print(
    f"Throughput range: "
    f"{summary['tokens_per_sec'].min():.2f} - "
    f"{summary['tokens_per_sec'].max():.2f} tokens/sec"
)

print(
    f"Average CPU temperature: "
    f"{summary['avg_cpu_temp_c'].mean():.2f} °C"
)

print(
    f"Average effective clock: "
    f"{summary['avg_effective_clock_mhz'].mean():.2f} MHz"
)

print(
    f"Clock ↔ throughput correlation: "
    f"{clock_correlation:.3f}"
)

print(
    f"Temperature ↔ throughput correlation: "
    f"{temperature_correlation:.3f}"
)


# --------------------------------------------------
# Graph 1: Run vs throughput
# --------------------------------------------------

plt.figure()

plt.plot(
    summary["run"],
    summary["tokens_per_sec"],
    marker="o"
)

plt.xlabel("Run")
plt.ylabel("Tokens/sec")
plt.title("LLM Throughput Across Runs")
plt.grid()

plt.savefig(
    "thermal/graphs/run_vs_throughput.png",
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Graph 2: Effective clock vs throughput
# --------------------------------------------------

plt.figure()

plt.scatter(
    summary["avg_effective_clock_mhz"],
    summary["tokens_per_sec"]
)

plt.xlabel("Average Effective Clock (MHz)")
plt.ylabel("Tokens/sec")
plt.title("Effective CPU Clock vs LLM Throughput")
plt.grid()

plt.savefig(
    "thermal/graphs/clock_vs_throughput.png",
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Graph 3: Temperature vs throughput
# --------------------------------------------------

plt.figure()

plt.scatter(
    summary["avg_cpu_temp_c"],
    summary["tokens_per_sec"]
)

plt.xlabel("Average CPU Temperature (°C)")
plt.ylabel("Tokens/sec")
plt.title("CPU Temperature vs LLM Throughput")
plt.grid()

plt.savefig(
    "thermal/graphs/temperature_vs_throughput.png",
    bbox_inches="tight"
)

plt.close()


print("\nAnalysis complete.")
print(f"Summary saved to: {SUMMARY_PATH}")