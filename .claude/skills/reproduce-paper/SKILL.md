---
description: Reproduce all paper results end-to-end. Runs zero-shot experiments for all 24 models, BM25 baselines, fine-tuning grid, and generates analysis tables and figures. Use when someone wants to fully reproduce the IdioLink paper.
---

## Reproduce Paper Results

Full paper reproduction workflow.

### Instructions

Follow these steps in order. Report progress after each major step.

#### Step 1: BM25 Baselines
```bash
python run_bm25.py --query_mode sentence --tune
python run_bm25.py --query_mode span --tune
```
Report: BM25 metrics for both modes.

#### Step 2: Zero-Shot Dense Retrieval (all 24 models × 4 modes)
```bash
python run_all.py
```
This is the longest step. Report: number of successful model×mode combinations.

#### Step 3: Fine-Tuning (5 models × 4 modes × 3 seeds)
Run for each of the 5 fine-tuning models:
```bash
python run_fine_tune.py --model sentence-transformers/all-MiniLM-L6-v2 --mode sentence --seeds 42 43 44
python run_fine_tune.py --model sentence-transformers/all-MiniLM-L6-v2 --mode span --seeds 42 43 44
python run_fine_tune.py --model sentence-transformers/all-MiniLM-L6-v2 --mode instruction_sentence --seeds 42 43 44
python run_fine_tune.py --model sentence-transformers/all-MiniLM-L6-v2 --mode instruction_span --seeds 42 43 44
```
Repeat for: `facebook/drama-1b`, `intfloat/e5-base-v2`, `BAAI/bge-m3`, `Qwen/Qwen3-Embedding-0.6B`.

#### Step 4: Generate Analysis Tables and Figures
```bash
python analysis/generate_zero_shot_table.py
python analysis/generate_finetuning_table.py
python analysis/generate_dataset_stats.py
python analysis/plot_performance.py
```

#### Step 5: Report Final Summary
- Display the full results CSV
- Display fine-tuning table (mean±std)
- Note any failures that need investigation
- Point to `assets/` for generated figures

### Notes
- Full reproduction requires significant GPU resources (7B+ models need ≥24GB VRAM)
- Expected duration: several hours to days depending on hardware
- Fine-tuning step alone: ~2-4 hours per model on a single GPU
