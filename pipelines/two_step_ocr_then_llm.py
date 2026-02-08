"""Two-step OCR -> LLM -> JSON pipeline.

Step 1: GLM-OCR (vision model) transcribes the image to raw text.
Step 2: Qwen2.5-3B-Instruct (text-only LLM) extracts structured JSON from that text.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from pipelines.one_step_json import generate, load_model, _resolve_device
from prompts.glm_ocr_prompts import build_text_extraction_prompt

RAW_OCR_PROMPT = "Transcribe all visible text from this image exactly as it appears."
LLM_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"


def run(input_path: str, task: str, *, device: str = "auto") -> str:
    image_path = Path(input_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input not found: {image_path}")

    # ── Step 1: raw OCR transcription (GLM-OCR) ──
    processor, ocr_model = load_model(device)
    raw_text = generate(processor, ocr_model, image_path, RAW_OCR_PROMPT)

    print("--- Raw OCR Output ---")
    print(raw_text)
    print("----------------------\n")

    # Free the vision model before loading the LLM
    del processor, ocr_model
    gc.collect()

    # ── Step 2: structured JSON extraction (Qwen2.5-3B) ──
    device_map, torch_dtype = _resolve_device(device)

    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)

    messages = build_text_extraction_prompt(task, raw_text)

    pipe = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tokenizer,
    )

    output = pipe(
        messages,
        max_new_tokens=1024,
        return_full_text=False,
        temperature=0.0,
        do_sample=False,
    )

    result = output[0]["generated_text"]

    # Free the LLM
    del llm_model, tokenizer, pipe
    gc.collect()

    return result.strip() if isinstance(result, str) else result
