"""Run evaluation over a dataset."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from eval.metrics import compute_metrics
from pipelines.one_step_json import run as run_one_step
from pipelines.two_step_ocr_then_llm import run as run_two_step


def _iter_inputs(dataset_dir: Path) -> List[Path]:
    allowed = {".png", ".jpg", ".jpeg", ".pdf"}
    return sorted(
        [
            path
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in allowed
        ]
    )


def _load_gold(input_path: Path) -> Dict[str, Any] | None:
    gold_path = input_path.with_suffix(".json")
    if not gold_path.exists():
        return None
    return json.loads(gold_path.read_text(encoding="utf-8"))


def run_eval(dataset_dir: str, task: str, output_dir: str | None = None) -> Dict[str, Any]:
    dataset_path = Path(dataset_dir).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("eval") / "outputs" / timestamp
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    for input_path in _iter_inputs(dataset_path):
        gold = _load_gold(input_path)

        one_step_text = run_one_step(str(input_path), task)
        two_step_text = run_two_step(str(input_path), task)

        one_step_metrics = compute_metrics(one_step_text, gold)
        two_step_metrics = compute_metrics(two_step_text, gold)

        sample = {
            "input": str(input_path),
            "one_step": {
                "output": one_step_text,
                "metrics": one_step_metrics,
            },
            "two_step": {
                "output": two_step_text,
                "metrics": two_step_metrics,
            },
        }
        results.append(sample)

    summary = {
        "dataset": str(dataset_path),
        "task": task,
        "samples": results,
    }

    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary
