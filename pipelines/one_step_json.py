"""One-step OCR to JSON using GLM-OCR."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from prompts.glm_ocr_prompts import build_prompt

MODEL_PATH = "zai-org/GLM-OCR"


def run(input_path: str, task: str, *, max_new_tokens: int = 1024) -> str:
    image_path = Path(input_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input not found: {image_path}")

    prompt = build_prompt(task)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

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


if __name__ == "__main__":
    raise SystemExit(
        "Run via cli.py (e.g., `python cli.py run --method one_step --input <file> --task receipt`)."
    )
