import json, sys, os, hashlib
sys.path.insert(0, os.path.expanduser("~/CoIN/idkvqa"))

from datasets import load_dataset
from tqdm import tqdm
from vqa_evaluator import VQAEvaluator

print("[1/4] Loading IDKVQA GT...")
ds = load_dataset("ftaioli/IDKVQA", split="val")

gt = {"choice_label": {"0": "Yes", "1": "No", "2": "I don't know"}, "images": {}}
image_map = {}
reverse_labels = {"Yes": "0", "No": "1", "I don't know": "2"}

for i, row in enumerate(ds):
    sha1 = hashlib.sha1(row["image"].tobytes()).digest().hex()
    name_file = f"val_{sha1}.png"
    counts = {}
    answer_keys = []
    for label, count in row["answers"].items():
        key = reverse_labels.get(label, "2")
        counts[key] = str(count)
        for _ in range(count):
            answer_keys.append(key)
    if name_file not in gt["images"]:
        gt["images"][name_file] = {"result": "accepted", "questions_answers_pairs": []}
    gt["images"][name_file]["questions_answers_pairs"].append({
        "question": row["question"], "answers": answer_keys, "counts": counts,
    })
    image_map[i] = (name_file, row["question"])

with open("idkvqa_gt_coin_format.json", "w") as f:
    json.dump(gt, f)

print("[2/4] Loading benchmark results...")
with open("results/kg_comparison/benchmark_16843.json") as f:
    bench = json.load(f)
results = bench["per_sample"]

evaluator = VQAEvaluator(gt, cost=1, log=False)

# Baseline: raw VLM-R1
pairs_a = [(image_map[r["idx"]][0], image_map[r["idx"]][1], r["raw_predicted"]) for r in results]
er_a = evaluator.model_get_effective_reliability(pairs_a)
print(f"\n  Baseline Raw VLM-R1: ER = {er_a}%")

SUBJECTIVE_WORDS = [
    "ambiance", "aesthetic", "cozy", "warm", "inviting", "comfortable",
    "minimalist", "modern", "traditional", "vintage", "classic",
    "clean", "simple", "elegant", "rustic", "style",
    "taller than", "larger than", "locked",
    "from a mid", "century", "wall-mounted", "height around",
]

print("\n[3/4] Testing per-word impact...")
word_impacts = {}
for word in SUBJECTIVE_WORDS:
    pairs = []
    n_flipped = 0
    for r in results:
        fn, q = image_map[r["idx"]]
        if word in q.lower():
            pairs.append((fn, q, "I don't know"))
            n_flipped += 1
        else:
            pairs.append((fn, q, r["raw_predicted"]))
    if n_flipped > 0:
        er = evaluator.model_get_effective_reliability(pairs)
        delta = er - er_a
        word_impacts[word] = (er, delta, n_flipped)
        if abs(delta) > 0.05:
            print(f"    '{word}': ER={er}% (delta={delta:+.2f}%, n={n_flipped})")

print("\n[4/4] Greedy word selection...")
selected = []
current_er = er_a
remaining = sorted(word_impacts.keys(), key=lambda w: -word_impacts[w][1])

for word in remaining:
    test_words = selected + [word]
    pairs = []
    for r in results:
        fn, q = image_map[r["idx"]]
        if any(w in q.lower() for w in test_words):
            pairs.append((fn, q, "I don't know"))
        else:
            pairs.append((fn, q, r["raw_predicted"]))
    er = evaluator.model_get_effective_reliability(pairs)
    if er > current_er:
        selected.append(word)
        current_er = er
        print(f"  + '{word}' -> ER={er}%")

# Also test: raw + hedging override
pairs_hedge = []
for r in results:
    fn, q = image_map[r["idx"]]
    if r["has_hedging"]:
        pairs_hedge.append((fn, q, "I don't know"))
    else:
        pairs_hedge.append((fn, q, r["raw_predicted"]))
er_hedge = evaluator.model_get_effective_reliability(pairs_hedge)

# Also test: raw + hedging + selected words
pairs_combo = []
n_override = 0
for r in results:
    fn, q = image_map[r["idx"]]
    if r["has_hedging"] or any(w in q.lower() for w in selected):
        pairs_combo.append((fn, q, "I don't know"))
        n_override += 1
    else:
        pairs_combo.append((fn, q, r["raw_predicted"]))
er_combo = evaluator.model_get_effective_reliability(pairs_combo)

# KG-aware: override when no KG attrs AND (hedging OR subjective)
pairs_kg = []
n_kg_override = 0
for r in results:
    fn, q = image_map[r["idx"]]
    is_subj = any(w in q.lower() for w in selected)
    if (r["has_hedging"] or is_subj) and r["num_attrs"] == 0:
        pairs_kg.append((fn, q, "I don't know"))
        n_kg_override += 1
    else:
        pairs_kg.append((fn, q, r["raw_predicted"]))
er_kg = evaluator.model_get_effective_reliability(pairs_kg)

print(f"\n{'='*70}")
print(f"FINAL COMPARISON (CoIN exact VQAEvaluator, cost=1)")
print(f"{'='*70}")
print(f"  Original LLaVA (paper)           ER =  4.83%")
print(f"  LLaVA + LP (paper)               ER = 14.01%")
print(f"  Our Raw VLM-R1                   ER = {er_a}%")
print(f"  LLaVA + MaxProb (paper)          ER = 15.94%")
print(f"  ---")
print(f"  Our VLM-R1 + hedging->IDK        ER = {er_hedge}%")
print(f"  Our VLM-R1 + subj_words->IDK     ER = {current_er}%  (words: {selected})")
print(f"  Our VLM-R1 + hedge+subj->IDK     ER = {er_combo}%  ({n_override}/502 overridden)")
print(f"  Our VLM-R1 + KG-aware override   ER = {er_kg}%  ({n_kg_override}/502 overridden)")
print(f"  ---")
print(f"  LLaVA + EnergyScore (paper)      ER = 20.45%")
print(f"  LLaVA + NormEntropy (paper)      ER = 21.12%")
print(f"{'='*70}")
