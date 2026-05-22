"""LLM-based annotation pipeline for validating generated idiom variants.

Uses GPT-4o-mini with 3x majority vote for validation.
Validates both sentence quality and span correctness independently.

Requires: openai or agno + API key in environment.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = None
    Field = None

try:
    from agno.agent import Agent
    from agno.models.google import Gemini
except ImportError:
    Agent = None
    Gemini = None

from .annotation_prompts import ANNOTATION_PROMPT


# Pydantic models for structured output
if BaseModel is not None:

    class SentenceAnnotation(BaseModel):
        id: str = Field(..., description="Sentence ID")
        llm_sentence_valid: int = Field(..., description="1=valid, 0=invalid")
        llm_span_valid: int = Field(..., description="1=valid, 0=invalid")
        llm_sentence_correction: str = Field(default="")
        llm_span_correction: str = Field(default="")
        sentence_issues: str = Field(default="")
        span_issues: str = Field(default="")

    class GroupAnnotationOutput(BaseModel):
        idiom: str
        subject: str
        annotations: List[SentenceAnnotation]


def load_generated_data(json_file: str) -> List[dict]:
    """Load generated idiom variants data."""
    with open(json_file, "r") as f:
        return json.load(f)


def group_by_idiom_subject(data: List[dict]) -> Dict[tuple, List[dict]]:
    """Group sentences by (idiom, subject) pairs."""
    groups = defaultdict(list)
    for item in data:
        key = (item["idiom"], item["subject"])
        groups[key].append(item)
    return dict(groups)


def format_group_for_annotation(group: List[dict]) -> str:
    """Format a group of sentences for the annotation agent."""
    if not group:
        return ""

    idiom = group[0]["idiom"]
    subject = group[0]["subject"]

    output = f"Idiom: {idiom}\nSubject: {subject}\nNumber of sentences: {len(group)}\n\n"
    output += "Sentences to validate:\n\n"

    for item in group:
        output += f"ID: {item['id']}\n"
        output += f"Sentence: {item['sentence']}\n"
        output += f"Span: {item['span']}\n"
        output += f"Usage: {item['usage']}\n"
        output += "---\n\n"

    return output


def create_annotation_agent(model_name: str = "gemini-2.5-pro", api_key: Optional[str] = None):
    """Create annotation agent.

    Requires agno + Gemini API key.
    """
    if Agent is None or Gemini is None:
        raise ImportError("agno and google-generativeai packages are required")

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    model = Gemini(id=model_name, api_key=api_key)

    return Agent(
        model=model,
        output_schema=GroupAnnotationOutput,
        instructions=ANNOTATION_PROMPT,
        markdown=False,
    )


def merge_annotations(
    original_data: List[dict], annotations: Dict[str, "SentenceAnnotation"]
) -> List[dict]:
    """Merge annotation results back into original data."""
    annotated_data = []

    for item in original_data:
        item_copy = item.copy()
        annotation = annotations.get(item["id"])

        if annotation:
            item_copy["llm_sentence_valid"] = annotation.llm_sentence_valid
            item_copy["llm_span_valid"] = annotation.llm_span_valid
            item_copy["llm_sentence_correction"] = annotation.llm_sentence_correction
            item_copy["llm_span_correction"] = annotation.llm_span_correction
            item_copy["llm_sentence_issues"] = annotation.sentence_issues
            item_copy["llm_span_issues"] = annotation.span_issues
        else:
            item_copy["llm_sentence_valid"] = -1
            item_copy["llm_span_valid"] = -1
            item_copy["llm_sentence_correction"] = ""
            item_copy["llm_span_correction"] = ""
            item_copy["llm_sentence_issues"] = "Not annotated"
            item_copy["llm_span_issues"] = "Not annotated"

        annotated_data.append(item_copy)

    return annotated_data


def annotate_data(
    input_file: str,
    output_dir: str = "output",
    model_name: str = "gemini-2.5-pro",
    num_groups: Optional[int] = None,
) -> List[dict]:
    """Run annotation pipeline on generated data.

    Args:
        input_file: Path to JSON file with generated variants.
        output_dir: Directory for annotated output.
        model_name: LLM model for annotation.
        num_groups: Maximum groups to annotate (None = all).

    Returns:
        Annotated data list.
    """
    data = load_generated_data(input_file)
    groups = group_by_idiom_subject(data)

    print(f"Loaded {len(data)} sentences in {len(groups)} groups")

    agent = create_annotation_agent(model_name)
    all_annotations = {}
    failed_groups = []

    group_items = list(groups.items())
    if num_groups:
        group_items = group_items[:num_groups]

    for (idiom, subject), group_data in group_items:
        try:
            formatted = format_group_for_annotation(group_data)
            response = agent.run(formatted)
            output = response.content
            for annotation in output.annotations:
                all_annotations[annotation.id] = annotation
        except Exception as e:
            failed_groups.append({"idiom": idiom, "subject": subject, "error": str(e)})

    annotated_data = merge_annotations(data, all_annotations)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"annotated_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(annotated_data, f, indent=2)

    # Statistics
    total = len([a for a in annotated_data if a.get("llm_sentence_valid", -1) != -1])
    both_valid = len(
        [a for a in annotated_data if a.get("llm_sentence_valid") == 1 and a.get("llm_span_valid") == 1]
    )

    print(f"Annotated: {total}, Both valid: {both_valid} ({both_valid/total*100:.1f}% if total > 0)")
    if failed_groups:
        print(f"Failed groups: {len(failed_groups)}")
    print(f"Saved to: {output_file}")

    return annotated_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotate generated idiom variants")
    parser.add_argument("input_file", help="JSON file with generated variants")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--num_groups", type=int, default=None)
    args = parser.parse_args()

    annotate_data(args.input_file, args.output_dir, args.model, args.num_groups)
