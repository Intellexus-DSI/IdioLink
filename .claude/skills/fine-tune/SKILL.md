---
description: Run contrastive fine-tuning for a model on the IdioLink benchmark. Use when the user wants to train a model with InfoNCE loss. Example: /fine-tune e5-base-v2 sentence
---

## Fine-Tune

Fine-tune a model: $ARGUMENTS

### Instructions

1. **Parse arguments** — expect `<model> [mode] [seeds]`
   - `model`: HuggingFace model ID or shortname
   - `mode` (optional): `sentence`, `span`, `instruction_sentence`, or `instruction_span` (default: `sentence`)
   - `seeds` (optional): space-separated seed list (default: `42 43 44`)

2. **Resolve model shortname** using the registry in `idiolink/models/registry.py`.

3. **Verify model is in fine-tuning list** — only these 5 models are designed for fine-tuning:
   - `sentence-transformers/all-MiniLM-L6-v2`
   - `facebook/drama-1b`
   - `intfloat/e5-base-v2`
   - `BAAI/bge-m3`
   - `Qwen/Qwen3-Embedding-0.6B`
   
   If user picks a different model, warn them but proceed if they confirm.

4. **Run fine-tuning:**
   ```bash
   python run_fine_tune.py --model <model_id> --mode <mode> --seeds <seed1> <seed2> <seed3>
   ```

5. **Training protocol** (for reference):
   - Loss: InfoNCE, temperature=0.05
   - Optimizer: AdamW, lr=2e-5, linear warmup (100 steps)
   - Early stopping: patience=3 on validation nDCG@10
   - Batch size: 32

6. **Report results** — after training, read metrics from `results/fine_tuning/<model_slug>/<mode>/seed_*/metrics.json` and display mean±std of nDCG@10 and R-Precision across seeds.
