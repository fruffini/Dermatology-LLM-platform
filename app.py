import gc
import json
import os
from datetime import datetime

import gradio as gr
import torch
from peft import PeftModel
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    MllamaForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    pipeline,
)

OUTPUT_DIR = "outputs"
MODEL_ROOT = os.path.join("models", "VLM")
PRIVATE_TOKEN = os.getenv("VLM_PRIVATE_TOKEN", "")
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

# Use the second GPU (0-based index) when available.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

MODEL_REGISTRY = {
    "PaliGemma Derm": {
        "model_id": "brucewayne0459/paligemma_derm",
        "base_prompt": "<image>\nIdentify the skin condition and describe key visual findings.",
        "requires_image_token": True,
        "model_type": "paligemma",
        "model_dir": os.path.join(MODEL_ROOT, "paligemma_derm"),
        "card_path": os.path.join(MODEL_ROOT, "paligemma_derm", "model_card.md"),
    },
    "DermatoLLama": {
        "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "adapter_id": "DermaVLM/DermatoLLama-full",
        "base_prompt": (
            "Analyze the dermatological condition shown in the image and provide a detailed report "
            "including body location."
        ),
        "requires_image_token": False,
        "model_type": "mllama_lora",
        "model_dir": os.path.join(MODEL_ROOT, "dermatollama_full"),
        "card_path": os.path.join(MODEL_ROOT, "dermatollama_full", "model_card.md"),
    },
    "MedGemma 1.5": {
        "model_id": "google/medgemma-1.5-4b-it",
        "base_prompt": "You are a dermatologist. Identify the skin condition and describe key visual findings.",
        "requires_image_token": False,
        "model_type": "medgemma_it",
    },
    "GPT-OSS 120B": {
        "model_id": "openai/gpt-oss-120b",
        "base_prompt": "Explain the finding clearly and concisely.",
        "requires_image_token": False,
        "model_type": "text_pipeline",
    },
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
    model_type = info["model_type"]
    if model_type == "paligemma":
        model_id = info["model_id"]
        processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, device_map={"": 0}, token=HF_TOKEN
        )
        model.eval()
    elif model_type == "medgemma_it":
        model_id = info["model_id"]
        processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN
        )
        model.eval()
    elif model_type == "text_pipeline":
        model_id = info["model_id"]
        text_pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
            token=HF_TOKEN,
        )
        _MODEL_CACHE[model_name] = (text_pipe, None)
        return text_pipe, None
    else:
        if not HF_TOKEN:
            raise RuntimeError(
                "Missing Hugging Face token. Set HUGGINGFACE_HUB_TOKEN (or HF_TOKEN) to access gated models."
            )
        base_model_name = info["model_id"]
        processor = AutoProcessor.from_pretrained(base_model_name, token=HF_TOKEN)
        model = MllamaForConditionalGeneration.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN
        )
        model = PeftModel.from_pretrained(model, info["adapter_id"], token=HF_TOKEN)
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


def _unload_other_models(selected_name: str) -> None:
    to_delete = [name for name in _MODEL_CACHE if name != selected_name]
    for name in to_delete:
        processor, model = _MODEL_CACHE.pop(name)
        del processor
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def run_inference(
    model_name: str,
    image_path: str,
    prompt: str,
    save_result: bool,
    save_format: str,
):
    model_type = MODEL_REGISTRY[model_name]["model_type"]
    if not image_path and model_type != "text_pipeline":
        return "Please upload an image before running inference.", None

    prompt = prompt or ""
    if MODEL_REGISTRY[model_name]["requires_image_token"] and "<image>" not in prompt:
        prompt = f"<image>\n{prompt}".strip()
    image_info = _image_summary(image_path) if image_path else "no image"
    image_name = os.path.basename(image_path) if image_path else "no_image"
    base_name, _ = os.path.splitext(image_name)
    safe_folder = _safe_slug(base_name) or "request"

    _unload_other_models(model_name)
    processor, model = _load_model(model_name)
    input_image = None
    if model_type != "text_pipeline":
        with Image.open(image_path) as input_image_handle:
            input_image = input_image_handle.convert("RGB")

    if model_type == "paligemma":
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
    elif model_type == "medgemma_it":
        messages = []
        if prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": prompt}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image."},
                    {"type": "image", "image": input_image},
                ],
            }
        )
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
            )
        generated_tokens = outputs[0][input_length:]
        decoded_output = processor.decode(generated_tokens, skip_special_tokens=True)
    elif model_type == "text_pipeline":
        message_text = prompt or "Explain the finding clearly and concisely."
        messages = [{"role": "user", "content": message_text}]
        outputs = processor(
            messages,
            max_new_tokens=256,
        )
        generated_text = outputs[0].get("generated_text", "")
        if isinstance(generated_text, list):
            last_item = generated_text[-1] if generated_text else ""
            if isinstance(last_item, dict):
                decoded_output = last_item.get("content", str(last_item))
            else:
                decoded_output = str(last_item)
        else:
            decoded_output = str(generated_text)
    else:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        input_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(
            images=input_image,
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)
        generation_config = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
        }
        input_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config,
                pad_token_id=(
                    processor.tokenizer.pad_token_id
                    if processor.tokenizer.pad_token_id is not None
                    else processor.tokenizer.eos_token_id
                ),
            )
        generated_tokens = outputs[0][input_length:]
        decoded_output = processor.decode(generated_tokens, skip_special_tokens=True)

    extra_note = ""
    if model_type == "text_pipeline" and image_path:
        extra_note = "\nNote: image input is ignored by this text-only model."
    response = (
        f"Model: {model_name}\n"
        f"Prompt: {prompt}\n"
        f"Image: {image_info}\n\n"
        f"Result:\n{decoded_output}{extra_note}"
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
        "Upload a dermatology image, pick a model, and write a prompt."
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
    def _on_model_change(name: str):
        _unload_other_models(name)
        return (
            MODEL_REGISTRY[name]["base_prompt"],
            f"**Suggested prompt**: {MODEL_REGISTRY[name]['base_prompt']}",
            _load_model_card(name),
        )

    model_choice.change(
        _on_model_change,
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
