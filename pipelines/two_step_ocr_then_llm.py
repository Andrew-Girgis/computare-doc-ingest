"""Two-step OCR -> LLM -> JSON pipeline."""

from __future__ import annotations

from pipelines.one_step_json import run as one_step_run


def run(input_path: str, task: str) -> str:
    """
    Placeholder two-step pipeline.

    TODO: Implement real OCR->text/markdown and LLM->JSON flow.
    For now, it delegates to the one-step pipeline to keep the CLI usable.
    """

    return one_step_run(input_path, task)
