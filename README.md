# ENSTA-AI-Project-Group11-2025-2026

This repository brings together the two fronts of the AI project from **Group 11 — ENSTA Paris, 2025-2026**.

---

## Projects

### 1. [AIUTA-VLM-R1 — Efficient Instance Navigation with Knowledge Graphs](AIUTA-VLM-R1-main/README.md)

Reimplementation of the AIUTA embodied navigation system, replacing the original multi-model approach (LLaVA 7B + GPT-4o, 10-14 API calls per detection) with a single 3B-parameter Vision Language Model (VLM-R1) coupled with a deterministic knowledge graph. The result: comparable performance (φ₁=21.71 vs 21.12) with a 2.3× smaller model and zero API cost.

### 2. [CoIN-Bench Evaluation — Benchmarking and Improving Results via Prompt Engineering](AI-Project---Improving-CoIN--main/EVAL_README.md)

Evaluation pipeline for the CoIN-Bench benchmark, focused on improving results through prompt engineering techniques. The system orchestrates vision and language models (Qwen2.5-Coder, LLaMA-3, GroundingDINO, LLaVA-NeXT) distributed across four GPUs via SLURM, all running locally at zero API cost.
