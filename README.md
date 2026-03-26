# ENSTA-AI-Project-Group11-2025-2026

This repository brings together the two fronts of the AI project from **Group 11 — ENSTA Paris, 2025-2026**.

---

## Projects

### 1. [AIUTA-VLM-R1 — Efficient Instance Navigation with Knowledge Graphs](AIUTA-VLM-R1-main/)

Reimplementation of the AIUTA embodied navigation system, replacing the original multi-model approach (LLaVA 7B + GPT-4o, 10-14 API calls per detection) with a single 3B-parameter Vision Language Model (VLM-R1) coupled with a deterministic knowledge graph. The result: comparable performance (φ₁=21.71 vs 21.12) with a 2.3× smaller model and zero API cost.

### 2. [CoIN-Bench Evaluation — Benchmarking and Improving Results via Prompt Engineering](AI-Project---Improving-CoIN--main/)

Evaluation pipeline for the CoIN-Bench benchmark, focused on improving results through prompt engineering techniques. The system orchestrates vision and language models (Qwen2.5-Coder, LLaMA-3, GroundingDINO, LLaVA-NeXT) distributed across four GPUs via SLURM, all running locally at zero API cost.

### 3. [Fine-tuning Experiments — LLaVA QLoRA on IDKVQA & COCO](AI-Project---Improving-CoIN--main/finetuning_experiments/)

Fine-tuning of LLaVA-1.6 (Mistral-7B) using QLoRA (4-bit) on a mix of COCO/VQAv2 (general VQA) and IDKVQA (task-specific uncertainty-aware yes/no/idk) datasets. The pipeline includes data augmentation (text template variations + geometric/photometric image transforms), image-hash-based splits to prevent data leakage, soft labels from human vote distributions, cosine scheduling with early stopping, and tau search for Effective Reliability (ER) evaluation as defined in the AIUTA paper. Contains both a standalone Python script and a Jupyter notebook.

---

## Documentation

For detailed information on each project, please refer to their respective READMEs:

| Project | README Location |
|---------|----------------|
| AIUTA-VLM-R1 | [`AIUTA-VLM-R1-main/README.md`](AIUTA-VLM-R1-main/README.md) |
| CoIN-Bench Evaluation | [`AI-Project---Improving-CoIN--main/EVAL_README.md`](AI-Project---Improving-CoIN--main/EVAL_README.md) |
| Fine-tuning Experiments (HouseVQA & COCO) | [`AI-Project---Improving-CoIN--main/finetuning_experiments/README.md`](AI-Project---Improving-CoIN--main/finetuning_experiments/README.md) |
