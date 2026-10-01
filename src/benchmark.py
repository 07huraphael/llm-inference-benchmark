import time
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32
)

model = model.to(device)
model.eval()

messages = [
    {
        "role": "user",
        "content": "Explain what an operating system is in three sentences."
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

latencies = []
tokens_per_second = []

# Warm-up
print("Warming up...")

with torch.inference_mode():
    model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False
    )

print("\nBenchmark started.")

for i in range(5):

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

    end = time.perf_counter()

    generated_tokens = (
        outputs.shape[1]
        - inputs["input_ids"].shape[1]
    )

    latency = end - start
    speed = generated_tokens / latency

    latencies.append(latency)
    tokens_per_second.append(speed)

    print(
        f"Run {i + 1}: "
        f"{latency:.3f} sec, "
        f"{speed:.2f} tokens/sec"
    )

print("\n--- Benchmark Result ---")

print(
    f"Average latency: "
    f"{statistics.mean(latencies):.3f} sec"
)

print(
    f"Average tokens/sec: "
    f"{statistics.mean(tokens_per_second):.2f}"
)

print(
    f"Min latency: "
    f"{min(latencies):.3f} sec"
)

print(
    f"Max latency: "
    f"{max(latencies):.3f} sec"
)