import pandas as pd
import matplotlib.pyplot as plt


# CSV 불러오기
df = pd.read_csv("results/output_length_benchmark.csv")


# 출력 길이별 평균 계산
summary = (
    df.groupby("output_length")
    [["latency", "tokens_per_sec"]]
    .mean()
    .reset_index()
)

print(summary)


# ---------------------------
# 그래프 1: 출력 길이 vs Latency
# ---------------------------

plt.figure()

plt.plot(
    summary["output_length"],
    summary["latency"],
    marker="o"
)

plt.xlabel("Output Length (tokens)")
plt.ylabel("Average Latency (seconds)")
plt.title("Output Length vs Latency")

plt.grid()

plt.savefig(
    "graphs/output_length_vs_latency.png",
    bbox_inches="tight"
)

plt.show()


# ---------------------------
# 그래프 2: 출력 길이 vs Tokens/sec
# ---------------------------

plt.figure()

plt.plot(
    summary["output_length"],
    summary["tokens_per_sec"],
    marker="o"
)

plt.xlabel("Output Length (tokens)")
plt.ylabel("Average Tokens/sec")
plt.title("Output Length vs Generation Speed")

plt.grid()

plt.savefig(
    "graphs/output_length_vs_speed.png",
    bbox_inches="tight"
)

plt.show()