import time
import csv
import statistics
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

OUTPUT_LENGTHS = [16, 32, 64, 128, 256]
REPEATS = 3

device = "cpu"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32
)

model = model.to(device)
model.eval()


messages = [
    {
        "role": "user",
        "content": "Explain what an operating system is."
    }
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


inputs = tokenizer(
    text,
    return_tensors="pt"
).to(device)


# -----------------------------------
# Warm-up
# -----------------------------------

print("\nWarming up...")

with torch.inference_mode():
    model.generate(
        **inputs,
        max_new_tokens=16,
        min_new_tokens=16,
        do_sample=False
    )


# 결과를 저장할 리스트
results = []


# -----------------------------------
# Benchmark
# -----------------------------------

for output_length in OUTPUT_LENGTHS:

    print(f"\n===== {output_length} tokens =====")

    latencies = []
    speeds = []

    for run in range(REPEATS):

        start = time.perf_counter()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=output_length,
                min_new_tokens=output_length,
                do_sample=False
            )

        end = time.perf_counter()

        generated_tokens = (
            outputs.shape[1]
            - inputs["input_ids"].shape[1]
        )

        latency = end - start
        tokens_per_sec = generated_tokens / latency

        latencies.append(latency)
        speeds.append(tokens_per_sec)

        print(
            f"Run {run + 1}: "
            f"{generated_tokens} tokens, "
            f"{latency:.3f} sec, "
            f"{tokens_per_sec:.2f} tokens/sec"
        )


        results.append({
            "output_length": output_length,
            "run": run + 1,
            "generated_tokens": generated_tokens,
            "latency": latency,
            "tokens_per_sec": tokens_per_sec
        })


    print(
        f"Average: "
        f"{statistics.mean(latencies):.3f} sec, "
        f"{statistics.mean(speeds):.2f} tokens/sec"
    )


# -----------------------------------
# CSV 저장
# -----------------------------------

csv_path = "results/output_length_benchmark.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "output_length",
            "run",
            "generated_tokens",
            "latency",
            "tokens_per_sec"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print("\nBenchmark finished!")
print(f"Results saved to: {csv_path}")