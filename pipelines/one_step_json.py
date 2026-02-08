"""One-step OCR to JSON using GLM-OCR."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from prompts.glm_ocr_prompts import build_prompt

MODEL_PATH = "zai-org/GLM-OCR"


def _resolve_device(device: str) -> tuple[str, torch.dtype | str]:
    value = device.strip().lower()
    if value == "cpu":
        return "cpu", torch.float32
    if value == "mps":
        return "mps", "auto"
    return "auto", "auto"


def load_model(device: str = "auto"):
    """Load the GLM-OCR processor and model (expensive, call once)."""
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    device_map, torch_dtype = _resolve_device(device)
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    return processor, model


def generate(processor, model, image_path: Path, prompt: str, *, max_new_tokens: int = 1024) -> str:
    """Run a single inference pass on an image with the given prompt."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    return output_text.strip()


def run(
    input_path: str,
    task: str,
    *,
    max_new_tokens: int = 1024,
    device: str = "auto",
) -> str:
    image_path = Path(input_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input not found: {image_path}")

    processor, model = load_model(device)
    prompt = build_prompt(task)
    return generate(processor, model, image_path, prompt, max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    raise SystemExit(
        "Run via cli.py (e.g., `python cli.py run --method one_step --input <file> --task receipt`)."
    )
