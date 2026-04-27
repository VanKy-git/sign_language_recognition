from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os
from pathlib import Path

BASE = str(Path(__file__).resolve().parents[1])
MODEL_NEW = f"{BASE}/model/t5_sign_model"

# Load checkpoint epoch 3 (cai da luu)
checkpoints = [d for d in os.listdir(MODEL_NEW) if d.startswith("checkpoint")]
checkpoints.sort()
print(f"Checkpoints co: {checkpoints}")

if checkpoints:
    best_ckpt = f"{MODEL_NEW}/{checkpoints[-1]}"
else:
    best_ckpt = MODEL_NEW

print(f"Load: {best_ckpt}")

tokenizer = T5Tokenizer.from_pretrained(best_ckpt)
model = T5ForConditionalGeneration.from_pretrained(best_ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def predict(keywords):
    input_text = f"keywords to sentence: {keywords}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=64, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


tests = [
    # ===== CO BAN =====
    "me cold need medicine",
]

print("\n=== KET QUA ===")
for kw in tests:
    print(f"  {kw:30} -> {predict(kw)}")
