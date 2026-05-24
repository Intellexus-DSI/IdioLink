"""Keyword (content-word) overlap between queries and their expected (relevant) docs.

Definition
----------
- Tokens: lowercased alphabetic words (regex `[a-z]+`, len >= 2).
- Keywords: tokens with English stopwords removed (sklearn ENGLISH_STOP_WORDS).
- Two variants per query, controlled by `--strip-span`:
    keep_span : keywords from the full sentence (query and doc).
    no_span   : idiom span string removed (case-insensitive) from both sides
                before tokenization. Isolates *context* overlap, since literal
                queries and `literal` / `idiomatic` docs share the PIE by
                construction.
- Per (query, doc) pair, two metrics on keyword sets Q, D:
    Jaccard       = |Q ∩ D| / |Q ∪ D|
    QueryRecall   = |Q ∩ D| / |Q|   (share of query keywords present in doc)
- Per query: mean across the query's expected (relevant) docs, per relevance rule.
- Report: aggregated mean over queries, grouped by usage bucket.
    ALL queries
    LITERAL queries  -> relevant: literal docs
    IDIOMATIC queries -> relevant: idiomatic+simplification+sense docs (combined)
    IDIOMATIC queries, by doc subtype: idiomatic / simplification / sense
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)
TOKEN_RE = re.compile(r"[a-z]+")


def keywords(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text.lower()) if len(t) >= 2 and t not in STOPWORDS}


def strip_span(text: str, span: str) -> str:
    if not span:
        return text
    return re.sub(re.escape(span), " ", text, flags=re.IGNORECASE)


def relevance_rule(query_usage: str) -> set[str]:
    if query_usage == "literal":
        return {"literal"}
    return {"idiomatic", "simplification", "sense"}


def load_split(split_dir: Path):
    with open(split_dir / "queries.json") as f:
        queries = json.load(f)
    with open(split_dir / "indexes.json") as f:
        docs = json.load(f)
    return queries, docs


def build_doc_index(docs):
    by_pie_usage = defaultdict(list)
    for d in docs:
        by_pie_usage[(d["idiom"], d["usage"])].append(d)
    return by_pie_usage


def overlap_for_query(query, docs, strip):
    if strip:
        q_text = strip_span(query["sentence"], query["span"])
    else:
        q_text = query["sentence"]
    q_kw = keywords(q_text)
    pair_scores = []
    for d in docs:
        if strip:
            d_text = strip_span(d["sentence"], d["span"])
        else:
            d_text = d["sentence"]
        d_kw = keywords(d_text)
        if not q_kw or not d_kw:
            jacc = 0.0
        else:
            inter = len(q_kw & d_kw)
            jacc = inter / len(q_kw | d_kw)
        qr = (len(q_kw & d_kw) / len(q_kw)) if q_kw else 0.0
        pair_scores.append((jacc, qr, len(q_kw), len(d_kw), len(q_kw & d_kw)))
    return q_kw, pair_scores


def summarize(values):
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def run(split_dir: Path, strip: bool):
    queries, docs = load_split(split_dir)
    by_pu = build_doc_index(docs)

    # Per-query mean overlap to its relevant docs (per relevance rule, combined).
    per_query_jacc_all = []
    per_query_qr_all = []
    per_query_jacc_by_qusage = defaultdict(list)
    per_query_qr_by_qusage = defaultdict(list)

    # For idiomatic queries: per-query mean to docs of each subtype.
    per_query_jacc_idiom_by_dsubtype = defaultdict(list)
    per_query_qr_idiom_by_dsubtype = defaultdict(list)

    # Sizes
    q_kw_sizes = []
    rel_doc_counts = []

    for q in queries:
        rel_usages = relevance_rule(q["usage"])
        rel_docs = []
        for u in rel_usages:
            rel_docs.extend(by_pu.get((q["idiom"], u), []))
        rel_doc_counts.append(len(rel_docs))
        if not rel_docs:
            continue
        q_kw, pairs = overlap_for_query(q, rel_docs, strip)
        q_kw_sizes.append(len(q_kw))
        if pairs:
            mean_jacc = statistics.mean(p[0] for p in pairs)
            mean_qr = statistics.mean(p[1] for p in pairs)
            per_query_jacc_all.append(mean_jacc)
            per_query_qr_all.append(mean_qr)
            per_query_jacc_by_qusage[q["usage"]].append(mean_jacc)
            per_query_qr_by_qusage[q["usage"]].append(mean_qr)

        if q["usage"] == "idiomatic":
            # Break down by doc subtype.
            for subtype in ("idiomatic", "simplification", "sense"):
                sub_docs = by_pu.get((q["idiom"], subtype), [])
                if not sub_docs:
                    continue
                _, sub_pairs = overlap_for_query(q, sub_docs, strip)
                per_query_jacc_idiom_by_dsubtype[subtype].append(
                    statistics.mean(p[0] for p in sub_pairs)
                )
                per_query_qr_idiom_by_dsubtype[subtype].append(
                    statistics.mean(p[1] for p in sub_pairs)
                )

    rows = []
    rows.append(("ALL queries", per_query_jacc_all, per_query_qr_all))
    for usage in ("literal", "idiomatic"):
        rows.append(
            (
                f"{usage.upper()} queries (combined relevant)",
                per_query_jacc_by_qusage[usage],
                per_query_qr_by_qusage[usage],
            )
        )
    for sub in ("idiomatic", "simplification", "sense"):
        rows.append(
            (
                f"  IDIOMATIC q -> {sub} docs",
                per_query_jacc_idiom_by_dsubtype[sub],
                per_query_qr_idiom_by_dsubtype[sub],
            )
        )

    return {
        "split_dir": str(split_dir),
        "strip_span": strip,
        "n_queries": len(queries),
        "n_docs": len(docs),
        "mean_query_keywords": statistics.mean(q_kw_sizes) if q_kw_sizes else 0.0,
        "mean_relevant_docs_per_query": statistics.mean(rel_doc_counts) if rel_doc_counts else 0.0,
        "rows": [
            {
                "bucket": name,
                "n_queries": len(jaccs),
                "jaccard_mean": summarize(jaccs)["mean"],
                "jaccard_median": summarize(jaccs)["median"],
                "qrecall_mean": summarize(qrs)["mean"],
                "qrecall_median": summarize(qrs)["median"],
            }
            for name, jaccs, qrs in rows
        ],
    }


def print_table(report):
    header = f"=== Lexical (keyword) overlap | split={report['split_dir']} | strip_span={report['strip_span']} ==="
    print(header)
    print(
        f"queries={report['n_queries']}  docs={report['n_docs']}  "
        f"mean_query_kw={report['mean_query_keywords']:.2f}  "
        f"mean_rel_docs/q={report['mean_relevant_docs_per_query']:.2f}"
    )
    print()
    col = f"{'Bucket':<42} {'#q':>5} {'Jacc μ':>8} {'Jacc med':>9} {'QR μ':>7} {'QR med':>7}"
    print(col)
    print("-" * len(col))
    for r in report["rows"]:
        print(
            f"{r['bucket']:<42} {r['n_queries']:>5} "
            f"{r['jaccard_mean']:>8.3f} {r['jaccard_median']:>9.3f} "
            f"{r['qrecall_mean']:>7.3f} {r['qrecall_median']:>7.3f}"
        )
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parent.parent / "data"),
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of table")
    args = ap.parse_args()

    split_dir = Path(args.data_root) / args.split
    reports = [run(split_dir, strip=False), run(split_dir, strip=True)]

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for rep in reports:
            print_table(rep)


if __name__ == "__main__":
    main()
