"""Prompt builders for OCR and extraction pipelines."""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("extract_schema.json")

TASK_RULES = {
    "bank_stmt": [
        "Dates must be in YYYY-MM-DD format.",
        "If a field is missing or unclear, return an empty string.",
    ],
    "receipt": [
        "transaction_date must be in YYYY-MM-DD format.",
        "card_last4 must be the last 4 digits of the payment card number (look for CARD NUMBER: *********1234 or similar masked patterns). Extract only the digits, ignore trailing letters.",
        "items is an array: include one object per line item found. Return as many items as appear on the receipt.",
        "If a field is missing or unclear, return an empty string.",
    ],
}


def load_schema(task: str) -> dict:
    data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if task not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(f"Unknown task '{task}'. Available: {available}")
    return data[task]


def build_prompt(task: str) -> str:
    schema = load_schema(task)
    schema_text = json.dumps(schema, indent=4, ensure_ascii=True)
    rules = [
        "Only these keys. No extra fields.",
        "All scalar values must be strings.",
    ] + TASK_RULES.get(task, [])
    rules_text = "\n".join(f"- {rule}" for rule in rules)

    return (
        "Please output only this JSON (no markdown, no extra text):\n"
        f"{schema_text}\n\n"
        "Rules:\n"
        f"{rules_text}"
    )


def build_text_extraction_prompt(task: str, raw_text: str) -> list[dict]:
    """Build chat messages for a text-only LLM to extract JSON from raw OCR text."""
    schema = load_schema(task)
    schema_text = json.dumps(schema, indent=4, ensure_ascii=True)
    rules = [
        "Include ALL keys from the schema, even if the value is unknown (use an empty string).",
        "No extra fields beyond what is shown.",
        "All scalar values must be strings.",
    ] + TASK_RULES.get(task, [])
    rules_text = "\n".join(f"- {rule}" for rule in rules)

    return [
        {
            "role": "system",
            "content": "You are a structured data extraction assistant. Output only valid JSON, no explanation.",
        },
        {
            "role": "user",
            "content": (
                "Here is raw text extracted from a document:\n\n"
                f"{raw_text}\n\n"
                "Extract the data into this exact JSON structure:\n"
                f"{schema_text}\n\n"
                "Rules:\n"
                f"{rules_text}"
            ),
        },
    ]
