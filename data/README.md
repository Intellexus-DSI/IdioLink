# IdioLink Benchmark Data

## Schema

Each split contains:
- `indexes.json` — Document corpus (indexed for retrieval)
- `queries.json` — Query set
- `triplets_*.jsonl` — Training/validation triplets (train/val only)

### Document fields (indexes.json)
| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique document identifier |
| sentence | string | Full sentence text |
| idiom | string | PIE (potentially idiomatic expression) |
| span | string | Idiom span within the sentence |
| subject | string | Subject domain (10 domains) |
| usage | string | `literal`, `idiomatic`, `simplification`, or `sense` |
| is_gold | bool | Whether human-verified |

### Query fields (queries.json)
| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique query identifier |
| sentence | string | Full query sentence |
| idiom | string | PIE |
| span | string | Idiom span within the query |
| subject | string | Subject domain |
| usage | string | `literal` or `idiomatic` |
| is_gold | bool | Whether human-verified |

## Relevance Rules

- **Literal query** → relevant docs = all *literal* docs for the same PIE
- **Idiomatic query** → relevant docs = all *idiomatic* + *simplification* + *sense* docs for the same PIE

## Split Statistics

| Split | PIEs | Documents | Queries |
|-------|------|-----------|---------|
| Train | 22 | 2,200 | 440 |
| Val | 10 | 1,000 | 200 |
| Test | 75 | 7,500 | 1,500 |
| **Total** | **107** | **10,700** | **2,140** |

## Evaluation Metrics

- **R-Precision**: Precision at R, where R = number of relevant docs for the query
- **nDCG@10**: Normalized Discounted Cumulative Gain at rank 10
