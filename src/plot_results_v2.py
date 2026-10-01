import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "results/output_length_benchmark_v2.csv"
)


summary = (
    df.groupby("output_length")
    .agg(
        mean_latency=("latency", "mean"),
        std_latency=("latency", "std"),
        mean_speed=("tokens_per_sec", "mean"),
        std_speed=("tokens_per_sec", "std")
    )
    .reset_index()
)


print(summary)


# -----------------------------
# Latency
# -----------------------------

plt.figure()

plt.errorbar(
    summary["output_length"],
    summary["mean_latency"],
    yerr=summary["std_latency"],
    marker="o",
    capsize=5
)

plt.xlabel("Output Length (tokens)")
plt.ylabel("Average Latency (seconds)")

plt.title(
    "Output Length vs Latency"
)

plt.grid()

plt.savefig(
    "graphs/output_length_vs_latency_v2.png",
    bbox_inches="tight"
)

plt.show()


# -----------------------------
# Generation speed
# -----------------------------

plt.figure()

plt.errorbar(
    summary["output_length"],
    summary["mean_speed"],
    yerr=summary["std_speed"],
    marker="o",
    capsize=5
)

plt.xlabel("Output Length (tokens)")
plt.ylabel("Average Tokens/sec")

plt.title(
    "Output Length vs Generation Speed"
)

plt.grid()

plt.savefig(
    "graphs/output_length_vs_speed_v2.png",
    bbox_inches="tight"
)

plt.show()