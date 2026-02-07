"""Command-line interface for document ingestion pipelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.run_eval import run_eval
from pipelines.one_step_json import run as run_one_step
from pipelines.two_step_ocr_then_llm import run as run_two_step


def _write_output(text: str, output_path: str | None) -> None:
    if output_path is None:
        print(text)
        return

    path = Path(output_path).expanduser().resolve()
    path.write_text(text, encoding="utf-8")
    print(f"Wrote output to {path}")


def run_command(args: argparse.Namespace) -> None:
    if args.method == "one_step":
        text = run_one_step(args.input, args.task)
    elif args.method == "two_step":
        text = run_two_step(args.input, args.task)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    _write_output(text, args.output)


def eval_command(args: argparse.Namespace) -> None:
    summary = run_eval(args.dataset, args.task, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a pipeline on a single input")
    run_parser.add_argument("--method", required=True, choices=["one_step", "two_step"])
    run_parser.add_argument("--input", required=True, help="Path to input document")
    run_parser.add_argument("--task", required=True, help="Task name (e.g., receipt, bank_stmt)")
    run_parser.add_argument("--output", help="Write output JSON to a file")
    run_parser.set_defaults(func=run_command)

    eval_parser = subparsers.add_parser("eval", help="Evaluate pipelines on a dataset")
    eval_parser.add_argument("--dataset", required=True, help="Dataset directory")
    eval_parser.add_argument("--task", required=True, help="Task name (e.g., receipt, bank_stmt)")
    eval_parser.add_argument("--output-dir", help="Directory for eval outputs")
    eval_parser.set_defaults(func=eval_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
