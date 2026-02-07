"""Evaluation metrics for JSON extraction."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Tuple


def _flatten(data: Any, prefix: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            yield from _flatten(value, new_prefix)
        return

    if isinstance(data, list):
        # Lists are hard to score without task-specific rules.
        # Represent them as a single joined string for now.
        joined = ", ".join(str(item) for item in data)
        yield (prefix, joined)
        return

    yield (prefix, "" if data is None else str(data))


def parse_json(text: str) -> Dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def field_accuracy(pred: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, float]:
    pred_flat = dict(_flatten(pred))
    gold_flat = dict(_flatten(gold))

    if not gold_flat:
        return {"field_accuracy": 0.0, "fields_total": 0, "fields_matched": 0}

    matched = 0
    for key, gold_val in gold_flat.items():
        pred_val = pred_flat.get(key, None)
        if pred_val == gold_val:
            matched += 1

    total = len(gold_flat)
    return {
        "field_accuracy": matched / total,
        "fields_total": total,
        "fields_matched": matched,
    }


def compute_metrics(pred_text: str, gold: Dict[str, Any] | None) -> Dict[str, Any]:
    pred_json = parse_json(pred_text)
    metrics = {
        "json_valid": pred_json is not None,
    }

    if gold is not None and pred_json is not None:
        metrics.update(field_accuracy(pred_json, gold))
    elif gold is not None:
        metrics.update({"field_accuracy": 0.0, "fields_total": len(gold), "fields_matched": 0})

    return metrics
