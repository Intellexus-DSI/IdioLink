"""Index ablation utilities: filter documents by usage type before indexing.

Two named presets:
  - lit_sim_sense: keep {literal, simplification, sense} (drop idiomatic)
  - lit_idiom:     keep {literal, idiomatic} (drop simplification + sense)

`--index_filter` CLI values accept either a preset name or a comma-separated
list of usage types (e.g. "literal,idiomatic") for ad-hoc exploration.
"""

from typing import Any, Dict, List, Set, Tuple

ABLATION_PRESETS: Dict[str, Set[str]] = {
    "lit_sim_sense": {"literal", "simplification", "sense"},
    "lit_idiom": {"literal", "idiomatic"},
}

VALID_USAGES: Set[str] = {"literal", "idiomatic", "simplification", "sense"}


def parse_index_filter(arg: str) -> Tuple[str, Set[str]]:
    """Parse --index_filter value into (slug, keep_set).

    Preset names are returned verbatim as the slug; explicit CSV lists are
    normalized to a sorted-join slug so the same set always lands in the same
    output directory.
    """
    if arg in ABLATION_PRESETS:
        return arg, set(ABLATION_PRESETS[arg])
    parts = [p.strip().lower() for p in arg.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --index_filter value")
    unknown = set(parts) - VALID_USAGES
    if unknown:
        raise ValueError(
            f"Unknown usage types in --index_filter: {sorted(unknown)}. "
            f"Valid: {sorted(VALID_USAGES)}"
        )
    keep = set(parts)
    slug = "_".join(sorted(keep))
    return slug, keep


def filter_docs_by_usage(
    sentences: List[str],
    metadata: List[Dict[str, Any]],
    keep: Set[str],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Filter parallel (sentences, metadata) lists by metadata['usage'] in keep."""
    pairs = [(s, m) for s, m in zip(sentences, metadata) if m.get("usage") in keep]
    if not pairs:
        return [], []
    sents, meta = zip(*pairs)
    return list(sents), list(meta)
