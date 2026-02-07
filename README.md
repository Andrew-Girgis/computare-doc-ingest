# Computare Doc Ingest

Lightweight CLI for document OCR and JSON extraction.

**Structure**
```
computare-doc-ingest/
├─ pipelines/
│  ├─ one_step_json.py
│  └─ two_step_ocr_then_llm.py
├─ prompts/
│  ├─ glm_ocr_prompts.py
│  └─ extract_schema.json
├─ eval/
│  ├─ dataset/
│  ├─ run_eval.py
│  └─ metrics.py
├─ app/
│  └─ demo.py
├─ cli.py
└─ README.md
```

**CLI**
```
# Run one file with one method
python cli.py run --method one_step --input sample.pdf --task bank_stmt
python cli.py run --method two_step --input sample.pdf --task bank_stmt

# Run evaluation on a set and compare
python cli.py eval --dataset eval/dataset --task bank_stmt
```

**Datasets**
- Place documents in `eval/dataset/`.
- If you have labels, add a JSON file with the same base name.
  Example: `eval/dataset/receipt_001.png` and `eval/dataset/receipt_001.json`.

**Tasks**
- Task schemas live in `prompts/extract_schema.json`.
- Update `prompts/glm_ocr_prompts.py` to adjust per-task rules.
