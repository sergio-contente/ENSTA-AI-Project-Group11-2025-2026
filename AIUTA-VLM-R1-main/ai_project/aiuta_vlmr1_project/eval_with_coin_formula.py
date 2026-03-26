"""
Re-evaluate our VLM-R1 results using CoIN's EXACT VQAEvaluator formula.

Their formula (from vqa_evaluator.py):
  - Model says Yes/No:
    - If k annotators agree: score = min(k/3, 1)
    - If 0 annotators agree: score = -cost (penalty)
  - Model says IDK: score = 0

This is different from our binary Phi formula.
"""

import json
import sys
import os
sys.path.insert(0, os.path.expanduser("~/CoIN/idkvqa"))

from datasets import load_dataset
from collections import defaultdict
import hashlib
from tqdm import tqdm

# Step 1: Generate ground truth in CoIN format
print("[1/3] Generating ground truth in CoIN format...")
ds = load_dataset("ftaioli/IDKVQA", split="val")

gt = {
    "choice_label": {"0": "Yes", "1": "No", "2": "I don't know"},
    "images": {}
}

image_map = {}  # idx -> filename for matching

for i, row in enumerate(tqdm(ds, desc="Building GT")):
    sha1 = hashlib.sha1(row["image"].tobytes()).digest().hex()
    name_file = f"val_{sha1}.png"
    
    answers_raw = row["answers"]
    
    # Build counts dict (key -> str count)
    reverse_labels = {"Yes": "0", "No": "1", "I don't know": "2"}
    counts = {}
    answer_keys = []
    for label, count in answers_raw.items():
        key = reverse_labels.get(label, "2")
        counts[key] = str(count)
        for _ in range(count):
            answer_keys.append(key)
    
    if name_file not in gt["images"]:
        gt["images"][name_file] = {
            "result": "accepted",
            "questions_answers_pairs": []
        }
    
    gt["images"][name_file]["questions_answers_pairs"].append({
        "question": row["question"],
        "answers": answer_keys,
        "counts": counts,
    })
    
    image_map[i] = (name_file, row["question"])

# Save GT
with open("idkvqa_gt_coin_format.json", "w") as f:
    json.dump(gt, f, indent=2)
print(f"  Saved GT with {len(gt['images'])} images")

# Step 2: Load our benchmark results
print("\n[2/3] Loading our benchmark results...")
bench_path = "results/kg_comparison/benchmark_16843.json"
with open(bench_path) as f:
    bench = json.load(f)

our_results = bench["per_sample"]
print(f"  Loaded {len(our_results)} samples")

# Step 3: Evaluate with CoIN's exact formula
print("\n[3/3] Evaluating with CoIN VQAEvaluator...")
from vqa_evaluator import VQAEvaluator

with open("idkvqa_gt_coin_format.json") as f:
    gt_data = json.load(f)

evaluator = VQAEvaluator(gt_data, cost=1, log=False)

# Build answer pairs for each config
configs = {
    "A_raw_vlmr1": "raw_predicted",
    "B_kg_strict": "kg_strict_predicted",
    "C_kg_medium": "kg_medium_predicted",
    "D_kg_hybrid": "kg_hybrid_predicted",
}

print(f"\n{'='*65}")
print(f"RESULTS WITH COIN's EXACT VQAEvaluator (cost=1)")
print(f"{'='*65}")

for config_name, pred_key in configs.items():
    pairs = []
    skipped = 0
    for r in our_results:
        idx = r["idx"]
        if idx not in image_map:
            skipped += 1
            continue
        filename, question = image_map[idx]
        answer = r[pred_key]
        pairs.append((filename, question, answer))
    
    try:
        er = evaluator.model_get_effective_reliability(pairs)
        print(f"  {config_name:<20} ER = {er}%")
    except Exception as e:
        print(f"  {config_name:<20} ERROR: {e}")

# Also evaluate with cost=0.5
evaluator_05 = VQAEvaluator(gt_data, cost=0.5, log=False)
print(f"\n{'='*65}")
print(f"RESULTS WITH COIN's EXACT VQAEvaluator (cost=0.5)")
print(f"{'='*65}")

for config_name, pred_key in configs.items():
    pairs = []
    for r in our_results:
        idx = r["idx"]
        if idx not in image_map:
            continue
        filename, question = image_map[idx]
        answer = r[pred_key]
        pairs.append((filename, question, answer))
    
    try:
        er = evaluator_05.model_get_effective_reliability(pairs)
        print(f"  {config_name:<20} ER = {er}%")
    except Exception as e:
        print(f"  {config_name:<20} ERROR: {e}")

print(f"\n{'='*65}")
print(f"REFERENCE (CoIN paper Table 5, LLaVA-v1.6-mistral-7b)")
print(f"{'='*65}")
print(f"  Original LLaVA        ER = 4.83%")
print(f"  + MaxProb             ER = 15.94%")
print(f"  + LP                  ER = 14.01%")
print(f"  + EnergyScore         ER = 20.45%")
print(f"  + NormalizedEntropy   ER = 21.12%")
print(f"{'='*65}")
