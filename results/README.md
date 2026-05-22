# Results Directory

This directory is populated by running experiments. It is gitignored except for this README.

## Structure

```
results/
├── zero_shot/{model_slug}/{mode}/metrics.json
├── bm25/{mode}/metrics.json
├── bm25/tuning_results.json
├── fine_tuning/{model_slug}/{mode}/seed_{N}/metrics.json
└── full_results.csv
```
