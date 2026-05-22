"""Error analysis for annotation quality assessment.

Compares LLM annotations against human annotations to identify:
- False positives: LLM says valid, human says invalid
- False negatives: LLM says invalid, human says valid
- Error patterns by usage type, idiom, and error category
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_split_data(filepath: str) -> List[dict]:
    """Load annotated split data."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_errors(data: List[dict]) -> Dict[str, List[dict]]:
    """Categorize items by error type (TP/TN/FP/FN).

    Requires both llm_overall_valid and human_valid fields.
    """
    categories = {
        "true_positives": [],
        "true_negatives": [],
        "false_positives": [],
        "false_negatives": [],
    }

    for item in data:
        error_type = item.get("error_type", "")
        if error_type.startswith("FP"):
            categories["false_positives"].append(item)
        elif error_type.startswith("FN"):
            categories["false_negatives"].append(item)
        elif error_type == "TP":
            categories["true_positives"].append(item)
        elif error_type == "TN":
            categories["true_negatives"].append(item)

    return categories


def analyze_false_positives(fps: List[dict]) -> dict:
    """Analyze patterns in false positives (LLM too permissive)."""
    analysis = {
        "total": len(fps),
        "by_category": Counter(),
        "by_usage": Counter(),
        "by_idiom": Counter(),
    }

    for fp in fps:
        analysis["by_category"][fp.get("error_type", "FP")] += 1
        analysis["by_usage"][fp["usage"]] += 1
        analysis["by_idiom"][fp["idiom"]] += 1

    return analysis


def analyze_false_negatives(fns: List[dict]) -> dict:
    """Analyze patterns in false negatives (LLM too strict)."""
    analysis = {
        "total": len(fns),
        "by_usage": Counter(),
        "by_idiom": Counter(),
        "llm_issue_patterns": Counter(),
    }

    for fn in fns:
        analysis["by_usage"][fn["usage"]] += 1
        analysis["by_idiom"][fn["idiom"]] += 1

        issues = fn.get("llm_issues", "").lower()
        if "usage label" in issues:
            analysis["llm_issue_patterns"]["Usage label overcorrection"] += 1
        elif "span" in issues:
            analysis["llm_issue_patterns"]["Span overcorrection"] += 1
        elif "grammar" in issues:
            analysis["llm_issue_patterns"]["Grammar hypercritical"] += 1
        else:
            analysis["llm_issue_patterns"]["Other"] += 1

    return analysis


def compute_metrics(categories: Dict[str, List[dict]]) -> dict:
    """Compute precision, recall, F1 from error categories."""
    tp = len(categories["true_positives"])
    tn = len(categories["true_negatives"])
    fp = len(categories["false_positives"])
    fn = len(categories["false_negatives"])

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def generate_report(data: List[dict]) -> str:
    """Generate a text report of error analysis."""
    categories = extract_errors(data)
    metrics = compute_metrics(categories)

    lines = ["Annotation Error Analysis Report", "=" * 50, ""]
    lines.append(f"Total samples: {sum(len(v) for v in categories.values())}")
    lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
    lines.append(f"Precision: {metrics['precision']:.4f}")
    lines.append(f"Recall: {metrics['recall']:.4f}")
    lines.append(f"F1: {metrics['f1']:.4f}")
    lines.append(f"TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    lines.append("")

    if categories["false_positives"]:
        fp_analysis = analyze_false_positives(categories["false_positives"])
        lines.append("False Positives (LLM too permissive):")
        lines.append(f"  Total: {fp_analysis['total']}")
        lines.append(f"  By usage: {dict(fp_analysis['by_usage'])}")
        lines.append(f"  By category: {dict(fp_analysis['by_category'])}")
        lines.append("")

    if categories["false_negatives"]:
        fn_analysis = analyze_false_negatives(categories["false_negatives"])
        lines.append("False Negatives (LLM too strict):")
        lines.append(f"  Total: {fn_analysis['total']}")
        lines.append(f"  By usage: {dict(fn_analysis['by_usage'])}")
        lines.append(f"  LLM issue patterns: {dict(fn_analysis['llm_issue_patterns'])}")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze annotation errors")
    parser.add_argument("input_file", help="JSON file with error_type field")
    args = parser.parse_args()

    data = load_split_data(args.input_file)
    report = generate_report(data)
    print(report)
