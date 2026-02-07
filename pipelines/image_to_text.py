import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

MODEL_PATH = "zai-org/GLM-OCR"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "25376111-7AE3-4903-BD84-FC0243C47BA6_1_105_c.jpeg"
            },
            {
                "type": "text",
                "text": "Please output only this JSON (no markdown, no extra text). Strings only:\n{\n    \"company_name\": \"\",\n    \"address\": \"\",\n    \"transaction_type\": \"\",\n    \"date\": \"\",\n    \"card_last4\": \"\",\n    \"items_purchased\": {\n        \"subtotal\": \"\",\n        \"total\": \"\",\n        \"approval_status\": \"\"\n    }\n}\n\nRules:\n- Only these keys. No extra fields.\n- transaction_type must be one of: purchase, sale, refund, other\n- items_purchased.approval_status must be one of: approved, declined\n- card_last4 must be last 4 digits of the card, or empty\n- date must be in format YYYY-MM-DD\n- If a field is missing or unclear, return an empty string"
            }
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
    return_tensors="pt"
).to(model.device)
inputs.pop("token_type_ids", None)
generated_ids = model.generate(**inputs, max_new_tokens=1024)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(output_text)


