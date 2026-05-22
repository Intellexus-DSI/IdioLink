---
description: Run a single retrieval experiment for a given model and query mode. Use when the user wants to evaluate one model on one configuration. Example: /run-experiment bge-m3 sentence
---

## Run Experiment

Run a single experiment for: $ARGUMENTS

### Instructions

1. **Parse arguments** — expect `<model> <query_mode>`
   - `model`: HuggingFace model ID or shortname (e.g., `bge-m3`, `e5-base-v2`, `BAAI/bge-m3`)
   - `query_mode`: one of `sentence`, `span`, `instruction_sentence`, `instruction_span`

2. **Resolve model shortname** — if user gave a shortname, resolve it by searching the registry in `idiolink/models/registry.py`. Common shortnames:
   - `sbert` → `sentence-transformers/all-MiniLM-L6-v2`
   - `bge-m3` → `BAAI/bge-m3`
   - `e5-base` → `intfloat/e5-base-v2`
   - `drama` → `facebook/drama-1b`
   - Otherwise, grep the registry for a match

3. **Select the correct runner** based on query_mode:
   - `sentence` or `span` → `python run_dense.py --model <model_id> --query_mode <mode>`
   - `instruction_sentence` or `instruction_span` → `python run_instruction.py --model <model_id> --query_mode <mode>`

4. **Run the experiment** and display the output.

5. **Show results** — read the output metrics file at `results/zero_shot/<model_slug>/<mode>/metrics.json` and display R-Precision and nDCG@10.

6. If the model fails to load (OOM, missing dependencies), suggest `/debug-model <model>` to diagnose.
