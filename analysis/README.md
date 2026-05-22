# Analysis Scripts

Scripts for generating result tables and figures from IdioLink experiments.

## Scripts

### `generate_zero_shot_table.py`

Produces a CSV and stdout table of zero-shot retrieval results.

- Reads from `results/zero_shot/{model_slug}/{mode}/metrics.json`
- Columns: model, then r_precision and nDCG@10 for each of 4 query modes
- Missing results shown as "-"

```bash
python analysis/generate_zero_shot_table.py
```

### `generate_finetuning_table.py`

Produces a table of fine-tuning results with mean +/- std across seeds.

- Reads from `results/fine_tuning/{model_slug}/{mode}/seed_{N}/metrics.json`
- Models and seeds defined in `config.yaml`

```bash
python analysis/generate_finetuning_table.py
```

### `generate_dataset_stats.py`

Reports dataset composition: PIE counts, document/query counts, usage distributions per split.

```bash
python analysis/generate_dataset_stats.py
```

### `plot_performance.py`

Generates matplotlib figures saved to `assets/`:

- `zero_shot_sentence_ndcg.png` - Bar chart of all models on sentence mode
- `top_models_by_mode.png` - Grouped bar of top-5 models across all 4 modes

```bash
python analysis/plot_performance.py
```

## Output

- CSV tables are saved to `assets/`
- Figures are saved to `assets/`
