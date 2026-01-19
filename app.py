import json
import os
from datetime import datetime

import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

OUTPUT_DIR = "outputs"
MODEL_ROOT = os.path.join("models", "VLM")
PRIVATE_TOKEN = os.getenv("VLM_PRIVATE_TOKEN", "")

# Use the second GPU (0-based index) when available.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

MODEL_REGISTRY = {
    "PaliGemma Derm": {
        "model_id": "brucewayne0459/paligemma_derm",
        "base_prompt": "Identify the skin condition and describe key visual findings.",
        "model_dir": os.path.join(MODEL_ROOT, "paligemma_derm"),
        "card_path": os.path.join(MODEL_ROOT, "paligemma_derm", "model_card.md"),
    }
}

_MODEL_CACHE = {}


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def _image_summary(image_path: str) -> str:
    if not image_path:
        return "no image"
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return f"{width}x{height}"
    except Exception:
        return "unknown"


def _load_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    info = MODEL_REGISTRY[model_name]
    model_id = info["model_id"]
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map={"": 0})
    model.eval()

    _MODEL_CACHE[model_name] = (processor, model)
    return processor, model


def _load_model_card(model_name: str) -> str:
    info = MODEL_REGISTRY.get(model_name, {})
    card_path = info.get("card_path")
    if card_path and os.path.exists(card_path):
        with open(card_path, "r", encoding="utf-8") as handle:
            return handle.read()
    return "No model card available."


def run_inference(
    model_name: str,
    image_path: str,
    prompt: str,
    save_result: bool,
    save_format: str,
):
    if not image_path:
        return "Please upload an image before running inference.", None

    prompt = prompt or ""
    image_info = _image_summary(image_path)
    image_name = os.path.basename(image_path)
    base_name, _ = os.path.splitext(image_name)
    safe_folder = _safe_slug(base_name) or "image"

    processor, model = _load_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with Image.open(image_path) as input_image:
        input_image = input_image.convert("RGB")
    inputs = processor(
        text=prompt,
        images=input_image,
        return_tensors="pt",
        padding="longest",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=80)
    input_length = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0][input_length:]
    decoded_output = processor.decode(generated_ids, skip_special_tokens=True)

    response = (
        f"Model: {model_name}\n"
        f"Prompt: {prompt}\n"
        f"Image: {image_info}\n\n"
        f"Result:\n{decoded_output}"
    )

    if not save_result:
        return response, None

    output_folder = os.path.join(OUTPUT_DIR, safe_folder)
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_model = _safe_slug(model_name) or "vlm"
    filename = f"{timestamp}_{safe_model}.{save_format}"
    path = os.path.join(output_folder, filename)

    if save_format == "json":
        payload = {
            "model": model_name,
            "model_id": MODEL_REGISTRY[model_name]["model_id"],
            "prompt": prompt,
            "image": image_info,
            "image_name": image_name,
            "result": response,
            "created_at_utc": timestamp,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(response + "\n")

    return response, path


with gr.Blocks(title="Dermatology VLM Inference") as demo:
    gr.Markdown(
        "# Dermatology VLM Inference\n"
        "Upload a dermatology image, pick a model, and provide a token."
    )

    with gr.Row():
        model_choice = gr.Dropdown(
            label="VLM model",
            choices=list(MODEL_REGISTRY.keys()),
            value="PaliGemma Derm",
        )
        prompt_input = gr.Textbox(
            label="Prompt",
            value=MODEL_REGISTRY["PaliGemma Derm"]["base_prompt"],
            placeholder="Enter prompt",
        )

    image_input = gr.Image(label="Dermatology image", type="filepath")

    base_prompt_note = gr.Markdown(
        f"**Suggested prompt**: {MODEL_REGISTRY['PaliGemma Derm']['base_prompt']}"
    )
    model_card = gr.Markdown(_load_model_card("PaliGemma Derm"))

    with gr.Row():
        save_result = gr.Checkbox(label="Save result", value=True)
        save_format = gr.Radio(
            label="Save format",
            choices=["txt", "json"],
            value="txt",
        )

    run_button = gr.Button("Run inference", variant="primary")
    clear_button = gr.Button("Clear")

    output_text = gr.Textbox(label="Result", lines=10)
    output_file = gr.File(label="Saved file")

    run_button.click(
        run_inference,
        inputs=[model_choice, image_input, prompt_input, save_result, save_format],
        outputs=[output_text, output_file],
    )
    model_choice.change(
        lambda name: (
            MODEL_REGISTRY[name]["base_prompt"],
            f"**Suggested prompt**: {MODEL_REGISTRY[name]['base_prompt']}",
            _load_model_card(name),
        ),
        inputs=[model_choice],
        outputs=[prompt_input, base_prompt_note, model_card],
    )
    clear_button.click(
        lambda: (None, MODEL_REGISTRY["PaliGemma Derm"]["base_prompt"], "PaliGemma Derm", True, "txt", "", None),
        outputs=[
            image_input,
            prompt_input,
            model_choice,
            save_result,
            save_format,
            output_text,
            output_file,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
