---
description: Diagnose model loading and encoding issues. Use when a model fails to load, crashes with OOM, or produces unexpected results. Example: /debug-model GritLM/GritLM-7B
---

## Debug Model

Diagnose issues with: $ARGUMENTS

### Instructions

1. **Parse argument** — expect a model ID (full HF path or shortname).

2. **Check registry** — verify the model exists in `idiolink/models/registry.py`:
   ```bash
   python -c "from idiolink.models.registry import MODEL_REGISTRY; print('$ARGUMENTS' in MODEL_REGISTRY or any('$ARGUMENTS' in k for k in MODEL_REGISTRY))"
   ```

3. **Check model config** — display the ModelConfig for this model:
   ```bash
   python -c "
   from idiolink.models.registry import MODEL_REGISTRY
   model_id = [k for k in MODEL_REGISTRY if '$ARGUMENTS' in k][0]
   cfg = MODEL_REGISTRY[model_id]
   print(f'Model: {cfg.model_id}')
   print(f'Class: {cfg.model_class}')
   print(f'Size: {cfg.size_params/1e9:.1f}B params')
   print(f'Trust remote code: {cfg.trust_remote_code}')
   print(f'Batch size: {cfg.batch_size}')
   print(f'Instruction format: {cfg.instruction_format}')
   "
   ```

4. **Attempt to load** — try loading the model and encoding a test sentence:
   ```bash
   python -c "
   from idiolink.models.registry import load_model
   from idiolink.utils import get_device
   device = get_device('auto')
   print(f'Device: {device}')
   model = load_model('<model_id>', device=device)
   print(f'Loaded OK. Embedding dim: {model.embedding_dim}')
   import numpy as np
   emb = model.encode(['The cat sat on the mat.'])
   print(f'Encoding OK. Shape: {emb.shape}, dtype: {emb.dtype}')
   print(f'Norm: {np.linalg.norm(emb[0]):.4f}')
   "
   ```

5. **Check GPU memory** (if CUDA):
   ```bash
   python -c "
   import torch
   if torch.cuda.is_available():
       print(f'GPU: {torch.cuda.get_device_name(0)}')
       print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
       print(f'Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
       print(f'Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB')
   else:
       print('No CUDA GPU available')
   "
   ```

6. **Diagnose common issues:**
   - OOM → suggest reducing batch_size or using a smaller model
   - `trust_remote_code` error → check if model needs `trust_remote_code=True`
   - Missing dependency (e.g., `gritlm` package) → suggest `pip install gritlm`
   - Tokenizer error → check transformers version compatibility

7. **Report findings** with a clear diagnosis and suggested fix.
