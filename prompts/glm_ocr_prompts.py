"""Prompt builders for GLM-OCR."""

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
        "card_last4 must be last 4 digits of the card, or empty.",
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
        "Strings only.",
    ] + TASK_RULES.get(task, [])
    rules_text = "\n".join(f"- {rule}" for rule in rules)

    return (
        "Please output only this JSON (no markdown, no extra text):\n"
        f"{schema_text}\n\n"
        "Rules:\n"
        f"{rules_text}"
    )
