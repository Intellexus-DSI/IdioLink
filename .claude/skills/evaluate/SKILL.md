---
description: View or regenerate evaluation metrics for existing experiment results. Use when the user wants to check results, compare models, or regenerate analysis tables. Example: /evaluate bge-m3 sentence
---

## Evaluate

Show evaluation results: $ARGUMENTS

### Instructions

1. **Parse arguments** — optional `<model> <mode>` or no args for full summary.

2. **If model+mode specified:**
   - Read `results/zero_shot/<model_slug>/<mode>/metrics.json`
   - Display R-Precision and nDCG@10
   - If file doesn't exist, inform user they need to run the experiment first

3. **If no arguments (full summary):**
   - Check if `results/full_results.csv` exists → display it as a table
   - If not, check what individual result files exist and compile a summary
   - Run `python analysis/generate_zero_shot_table.py` to regenerate the table

4. **For fine-tuning results:**
   - If user mentions "fine-tune" or "training", read from `results/fine_tuning/`
   - Run `python analysis/generate_finetuning_table.py` to regenerate

5. **For dataset stats:**
   - If user asks about data/dataset, run `python analysis/generate_dataset_stats.py`

6. **Display results** in a clean markdown table format.
