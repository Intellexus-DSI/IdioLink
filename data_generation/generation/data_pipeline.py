"""Idiom variant generation pipeline using Gemini API.

Generates sentence variants for idioms across subject domains:
- Literal: idiom words used in concrete/physical sense
- Idiomatic (figurative): standard metaphorical meaning
- Simplification: paraphrase without using the idiom
- Sense: explanation of the idiom's meaning

Requires: google-generativeai or agno + Gemini API key in environment.
"""

import csv
import json
import os
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

SUBJECTS = [
    "Politics",
    "Sport",
    "Technology",
    "History",
    "Medicine",
    "Culture",
    "Entertainment",
    "Food",
    "Business",
    "Environment",
]

SYSTEM_PROMPT_TEMPLATE = """You are an expert computational linguist specializing in idiomatic expressions.
Generate exactly {num_literal} literal, {num_figurative} figurative, {num_simplified} simplified figurative,
and {num_sense} sense-based figurative variants for the given idiom.

ALL variants must be within the subject domain: "{subject}"

Requirements:
- Literal: Use idiom words in physical/concrete non-idiomatic sense
- Figurative: Use idiom in standard idiomatic meaning
- Simplified: Reuse EXACT context from corresponding figurative variant, only replace the idiom
- Sense: Independent sentences explaining the idiom's meaning without using it

For each variant provide: sentence, span (exact idiom form or meaning phrase), subject.
Sentences must be 10-30 words, grammatically correct, and contextually clear.
"""


# Pydantic models for structured output (only available if pydantic is installed)
if BaseModel is not None:

    class SentenceVariant(BaseModel):
        sentence: str = Field(..., description="The generated sentence")
        span: str = Field(..., description="Idiom form or meaning-equivalent phrase")
        subject: str = Field(..., description="Subject domain")

    class VariantsOutput(BaseModel):
        idiom: str = Field(..., description="The original idiom")
        literal_variants: List[SentenceVariant]
        figurative_variants: List[SentenceVariant]
        simplified_figurative_variants: List[SentenceVariant]
        sense_figurative_variants: List[SentenceVariant]


def load_idioms_with_ids(csv_file: str) -> Dict[str, str]:
    """Load idioms from CSV with pre-assigned IDs. Only includes valid=1 idioms."""
    idiom_to_id = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("valid", "1") == "1":
                idiom_to_id[row["idiom"]] = row["id"]
    return idiom_to_id


def create_agent(
    subject: str,
    num_literal: int = 4,
    num_figurative: int = 2,
    num_simplified: int = 2,
    num_sense: int = 2,
    model_name: str = "gemini-2.5-pro",
    api_key: Optional[str] = None,
):
    """Create an Agno agent for variant generation.

    Requires agno and google-generativeai packages plus GEMINI_API_KEY env var.
    """
    if Agent is None or Gemini is None:
        raise ImportError("agno and google-generativeai packages are required")

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    model = Gemini(id=model_name, api_key=api_key)

    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        subject=subject,
        num_literal=num_literal,
        num_figurative=num_figurative,
        num_simplified=num_simplified,
        num_sense=num_sense,
    )

    return Agent(
        model=model,
        output_schema=VariantsOutput,
        instructions=prompt,
        markdown=False,
    )


def flatten_results(results: List[dict], idiom_to_id: Dict[str, str]) -> List[dict]:
    """Transform grouped results into flat sentence records with IDs.

    ID format: {idiom_id}_s{subject_idx:02d}_{usage}_{variant_idx:02d}
    """
    flat_data = []

    for result in results:
        base_idiom = result["idiom"]
        idiom_id = idiom_to_id.get(base_idiom, "000")
        subject_idx = result["subject_idx"]

        variant_types = [
            ("literal_variants", "lit"),
            ("figurative_variants", "idiom"),
            ("simplified_figurative_variants", "sim"),
            ("sense_figurative_variants", "sense"),
        ]

        usage_labels = {
            "lit": "literal",
            "idiom": "idiomatic",
            "sim": "simplification",
            "sense": "sense",
        }

        for field, prefix in variant_types:
            for var_idx, variant in enumerate(result.get(field, []), start=1):
                flat_data.append(
                    {
                        "id": f"{idiom_id}_s{subject_idx:02d}_{prefix}_{var_idx:02d}",
                        "sentence": variant["sentence"],
                        "idiom": base_idiom,
                        "span": variant["span"],
                        "subject": variant["subject"],
                        "usage": usage_labels[prefix],
                    }
                )

    return flat_data


def generate_variants(
    idioms_csv: str,
    output_dir: str = "output",
    subjects: Optional[List[str]] = None,
    num_literal: int = 4,
    num_figurative: int = 2,
    num_simplified: int = 2,
    num_sense: int = 2,
    model_name: str = "gemini-2.5-pro",
    max_idioms: Optional[int] = None,
    offset: int = 0,
) -> List[dict]:
    """Generate idiom variants for all idioms in the CSV.

    Args:
        idioms_csv: Path to CSV with id,idiom,valid columns.
        output_dir: Directory for output JSON files.
        subjects: List of subject domains (defaults to all 10).
        num_literal: Literal variants per subject per idiom.
        num_figurative: Figurative variants per subject per idiom.
        num_simplified: Simplified variants per subject per idiom.
        num_sense: Sense variants per subject per idiom.
        model_name: Gemini model ID.
        max_idioms: Maximum number of idioms to process.
        offset: Number of idioms to skip from the start.

    Returns:
        List of flat sentence records.
    """
    subjects = subjects or SUBJECTS
    idiom_to_id = load_idioms_with_ids(idioms_csv)
    all_idioms = list(idiom_to_id.keys())

    selected = all_idioms[offset:]
    if max_idioms:
        selected = selected[:max_idioms]

    os.makedirs(output_dir, exist_ok=True)

    results = []
    failed = []

    for idiom in selected:
        for subj_idx, subject in enumerate(subjects, start=1):
            try:
                agent = create_agent(
                    subject, num_literal, num_figurative, num_simplified, num_sense, model_name
                )
                response = agent.run(idiom)
                output = response.content

                results.append(
                    {
                        "idiom": output.idiom,
                        "subject": subject,
                        "subject_idx": subj_idx,
                        "literal_variants": [
                            {"sentence": v.sentence, "span": v.span, "subject": v.subject}
                            for v in output.literal_variants
                        ],
                        "figurative_variants": [
                            {"sentence": v.sentence, "span": v.span, "subject": v.subject}
                            for v in output.figurative_variants
                        ],
                        "simplified_figurative_variants": [
                            {"sentence": v.sentence, "span": v.span, "subject": v.subject}
                            for v in output.simplified_figurative_variants
                        ],
                        "sense_figurative_variants": [
                            {"sentence": v.sentence, "span": v.span, "subject": v.subject}
                            for v in output.sense_figurative_variants
                        ],
                    }
                )
            except Exception as e:
                failed.append({"idiom": idiom, "subject": subject, "error": str(e)})

    flat_data = flatten_results(results, idiom_to_id)

    # Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"idiom_variants_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(flat_data, f, indent=2)

    print(f"Generated {len(flat_data)} sentence records from {len(results)} batches")
    if failed:
        print(f"Failed: {len(failed)} batches")
    print(f"Saved to: {output_file}")

    return flat_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate idiom variants")
    parser.add_argument("idioms_csv", help="CSV file with id,idiom,valid columns")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--num_literal", type=int, default=4)
    parser.add_argument("--num_figurative", type=int, default=2)
    parser.add_argument("--num_simplified", type=int, default=2)
    parser.add_argument("--num_sense", type=int, default=2)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--max_idioms", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    generate_variants(
        args.idioms_csv,
        output_dir=args.output_dir,
        num_literal=args.num_literal,
        num_figurative=args.num_figurative,
        num_simplified=args.num_simplified,
        num_sense=args.num_sense,
        model_name=args.model,
        max_idioms=args.max_idioms,
        offset=args.offset,
    )
