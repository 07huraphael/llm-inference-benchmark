import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

device = "cpu"

print("Device:", device)
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

print("Generating...")

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

elapsed = end - start

generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\n--- Result ---")
print(generated_text)

print("\n--- Performance ---")
print(f"Generated tokens: {generated_tokens}")
print(f"Latency: {elapsed:.3f} seconds")
print(f"Tokens/sec: {generated_tokens / elapsed:.2f}")