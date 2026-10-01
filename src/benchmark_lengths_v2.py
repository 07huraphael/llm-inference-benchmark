import time
import csv
import random
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

OUTPUT_LENGTHS = [16, 32, 64, 128, 256]

REPEATS = 5

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


# -----------------------------
# Benchmark
# -----------------------------

for repeat in range(1, REPEATS + 1):

    print(f"\n===== Round {repeat} =====")

    test_order = OUTPUT_LENGTHS.copy()

    random.shuffle(test_order)

    print("Test order:", test_order)


    for position, output_length in enumerate(
        test_order,
        start=1
    ):

        # 잠깐 쉬어서 CPU가 계속 최대 부하가 되는 것을 완화
        time.sleep(2)

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

        tokens_per_sec = (
            generated_tokens / latency
        )


        print(
            f"{output_length:3d} tokens | "
            f"{latency:7.3f} sec | "
            f"{tokens_per_sec:6.2f} tokens/sec"
        )


        results.append({

            "round": repeat,

            "position": position,

            "output_length": output_length,

            "generated_tokens": generated_tokens,

            "latency": latency,

            "tokens_per_sec": tokens_per_sec
        })


# -----------------------------
# CSV 저장
# -----------------------------

csv_path = "results/output_length_benchmark_v2.csv"


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "round",
            "position",
            "output_length",
            "generated_tokens",
            "latency",
            "tokens_per_sec"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


print("\nBenchmark finished.")

print(
    f"Results saved to: {csv_path}"
)