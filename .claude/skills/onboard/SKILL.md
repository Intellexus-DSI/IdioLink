---
description: Set up the IdioLink project for a new user. Installs dependencies, verifies the environment, runs a smoke test, and explains the project structure. Use when someone is getting started with this repo for the first time.
---

## Onboarding

Help the user get started with the IdioLink project.

### Steps

1. **Check Python version** — must be 3.9+
   ```bash
   python --version
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify environment** — run a quick smoke test with 5 queries
   ```bash
   python run_dense.py --model sentence-transformers/all-MiniLM-L6-v2 --query_mode sentence --debug
   ```

4. **Explain project structure** to the user:
   - `run_dense.py` — dense retrieval (sentence/span modes)
   - `run_instruction.py` — instruction-aware retrieval (instruction_sentence/instruction_span)
   - `run_bm25.py` — BM25 lexical baseline
   - `run_fine_tune.py` — contrastive fine-tuning
   - `run_all.py` — full experiment matrix (24 models × 4 modes)
   - `idiolink/` — core library (models, evaluator, retriever, trainer)
   - `data/` — benchmark splits (train/val/test)
   - `analysis/` — scripts to generate paper tables and figures
   - `config.yaml` — central configuration

5. **Show available skills** — list the slash commands:
   - `/run-experiment` — run a single model+mode experiment
   - `/run-all` — run full experiment grid
   - `/fine-tune` — contrastive fine-tuning
   - `/evaluate` — view/recompute metrics
   - `/reproduce-paper` — full paper reproduction
   - `/debug-model` — diagnose model loading issues

6. **Confirm** that the smoke test produced metrics (R-Precision and nDCG@10) and report success.
