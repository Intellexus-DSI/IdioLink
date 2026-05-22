---
description: Run the full experiment matrix (all models × all modes) or a filtered subset. Use when the user wants to benchmark multiple models at once. Example: /run-all --debug
---

## Run All Experiments

Run the full experiment grid: $ARGUMENTS

### Instructions

1. **Parse arguments** — optional flags:
   - No args → run all 24 models × 4 modes on full test set
   - `--debug` → run all combinations on 5 queries only (smoke test)
   - Model names → filter to those models only (e.g., `bge-m3 e5-base-v2`)

2. **Build the command:**
   ```bash
   python run_all.py [--debug] [--models <model1> <model2> ...]
   ```

3. **Run it** — this may take a long time for the full grid. Inform the user of expected duration:
   - `--debug`: ~5-10 minutes (loads each model, encodes 5 queries)
   - Full run with small models only: ~30-60 minutes
   - Full run with all 24 models: several hours (7B+ models are slow)

4. **Report results** — after completion, read `results/full_results.csv` and display a summary table.

5. **Handle failures** — if some models fail, note which ones failed and suggest `/debug-model <model>` for each.
