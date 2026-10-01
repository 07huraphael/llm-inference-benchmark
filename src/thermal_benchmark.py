import time
import csv
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

OUTPUT_LENGTH = 128
REPEATS = 20
REST_SECONDS = 0

device = "cpu"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)


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


# -----------------------------
# Warm-up
# -----------------------------

print("\nWarming up...")

with torch.inference_mode():

    model.generate(
        **inputs,
        max_new_tokens=32,
        min_new_tokens=32,
        do_sample=False
    )


results = []


print("\nThermal benchmark started.")


# -----------------------------
# Benchmark
# -----------------------------

for run in range(1, REPEATS + 1):

    timestamp_start = datetime.now()

    start = time.perf_counter()


    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=OUTPUT_LENGTH,
            min_new_tokens=OUTPUT_LENGTH,
            do_sample=False
        )


    end = time.perf_counter()

    timestamp_end = datetime.now()


    generated_tokens = (
        outputs.shape[1]
        - inputs["input_ids"].shape[1]
    )


    latency = end - start

    tokens_per_sec = (
        generated_tokens / latency
    )


    print(
        f"Run {run:02d} | "
        f"{latency:7.3f} sec | "
        f"{tokens_per_sec:6.2f} tokens/sec"
    )


    results.append({

        "run": run,

        "start_time":
            timestamp_start.isoformat(
                timespec="seconds"
            ),

        "end_time":
            timestamp_end.isoformat(
                timespec="seconds"
            ),

        "generated_tokens":
            generated_tokens,

        "latency":
            latency,

        "tokens_per_sec":
            tokens_per_sec
    })


    time.sleep(
        REST_SECONDS
    )


# -----------------------------
# CSV 저장
# -----------------------------

csv_path = (
    "thermal/results/"
    "llm_thermal_benchmark.csv"
)


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "run",
            "start_time",
            "end_time",
            "generated_tokens",
            "latency",
            "tokens_per_sec"
        ]
    )

    writer.writeheader()

    writer.writerows(
        results
    )


print("\nBenchmark finished.")

print(
    f"Results saved to: {csv_path}"
)