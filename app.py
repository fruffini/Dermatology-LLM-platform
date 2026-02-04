import gc
import json
import os
from datetime import datetime

import gradio as gr
import torch
from peft import PeftModel
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    MllamaForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    TextIteratorStreamer,
    pipeline,
)
import threading

<<<<<<< HEAD




OUTPUT_DIR = "outputs"
OUTPUT_DIR = "outputs"
=======
OUTPUT_DIR = "outputs/baseline_vlm_ramo_prova_conflitto"
var = os.getenv("OUTPUT_DIR")
if var:
    OUTPUT_DIR = var
>>>>>>> 5205ce5 (correction to prova, back as before)
MODEL_ROOT = os.path.join("models", "VLM")
PRIVATE_TOKEN = os.getenv("VLM_PRIVATE_TOKEN", "")
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

# Use the second GPU (0-based index) when available.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

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
    "MedGemma 27B Text": {
        "model_id": os.getenv("MEDGEMMA_27B_MODEL_ID", "google/medgemma-27b-it"),
        "base_prompt": "Answer the multiple-choice question with only the best answer label.",
        "requires_image_token": False,
        "model_type": "medgemma_text_it",
    },
}

_MODEL_CACHE = {}

MCQ_PROMPT_TEMPLATE = (
    "You are a medical exam assistant. Answer the question using the choices provided.\n"
    "Return only the single best answer label (e.g., A, B, C) with no extra text.\n\n"
    "Question:\n{question}\n\n"
    "Choices:\n{choices}\n\n"
    "Answer:"
)
MCQ_MAX_NEW_TOKENS = int(os.getenv("MCQ_MAX_NEW_TOKENS", "8"))


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
    elif model_type == "medgemma_text_it":
        model_id = info["model_id"]
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN
        )
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, model)
        return tokenizer, model
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


def _format_choices(choices):
    lines = []
    for choice in choices or []:
        label = choice.get("label", "")
        text = choice.get("text", "")
        lines.append(f"{label}. {text}".strip())
    return "\n".join(lines)


def _load_mcq_file(file_obj):
    if not file_obj:
        return [], []
    path = getattr(file_obj, "name", None) or file_obj
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    items = data.get("items", [])
    options = []
    for item in items:
        item_id = item.get("id")
        question = item.get("question", "").replace("\n", " ")
        label = f"{item_id}: {question[:80]}".strip()
        options.append((label, item_id))
    return items, options


def _get_mcq_by_id(items, item_id):
    for item in items:
        if item.get("id") == item_id:
            return item
    return None


def _build_mcq_prompt(item):
    if not item:
        return ""
    choices = _format_choices(item.get("choices", []))
    return MCQ_PROMPT_TEMPLATE.format(question=item.get("question", ""), choices=choices)


def _generate_medgemma_text(tokenizer, model, message):
    messages = [{"role": "user", "content": message}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    input_length = input_ids.shape[-1]
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=MCQ_MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def _stream_medgemma_text(tokenizer, model, message):
    messages = [{"role": "user", "content": message}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": MCQ_MAX_NEW_TOKENS,
        "do_sample": False,
        "use_cache": True,
        "streamer": streamer,
    }
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    partial = ""
    for token_text in streamer:
        partial += token_text
        yield partial.strip()
    thread.join()


def _chat_with_model(history, message, model_name):
    if not message:
        return history, ""
    _unload_other_models(model_name)
    processor, model = _load_model(model_name)
    history = history or []
    model_type = MODEL_REGISTRY[model_name]["model_type"]
    if model_type == "medgemma_text_it":
        reply = _generate_medgemma_text(processor, model, message)
    elif model_type == "text_pipeline":
        convo_lines = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                convo_lines.append(f"User: {content}")
            elif role == "assistant":
                convo_lines.append(f"Assistant: {content}")
        convo_lines.append(f"User: {message}")
        convo_lines.append("Assistant:")
        prompt = "\n".join(convo_lines)
        outputs = processor(prompt, max_new_tokens=256)
    else:
        messages = history + [{"role": "user", "content": message}]
        outputs = processor(messages, max_new_tokens=256)
    generated_text = outputs[0].get("generated_text", "")
    if isinstance(generated_text, list):
        last_item = generated_text[-1] if generated_text else ""
        if isinstance(last_item, dict):
            reply = last_item.get("content", str(last_item))
        else:
            reply = str(last_item)
    else:
        reply = str(generated_text)
        if model_type == "text_pipeline":
            if reply.startswith(prompt):
                reply = reply[len(prompt):].strip()
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


def _chat_with_model_stream(history, message, model_name):
    if not message:
        yield history, ""
        return
    _unload_other_models(model_name)
    processor, model = _load_model(model_name)
    history = history or []
    model_type = MODEL_REGISTRY[model_name]["model_type"]
    if model_type == "medgemma_text_it":
        for partial in _stream_medgemma_text(processor, model, message):
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": partial},
            ], ""
        return
    new_history, _ = _chat_with_model(history, message, model_name)
    yield new_history, ""


with gr.Blocks(title="Dermatology VLM Inference") as demo:
    gr.Markdown(
        "# Dermatology VLM Inference\n"
        "Upload a dermatology image, pick a model, and write a prompt."
    )

    with gr.Tabs():
        with gr.Tab("VLM Inference"):
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

        with gr.Tab("MCQ Chat (MedGemma 27B Text)"):
            gr.Markdown(
                "Load a formatted MCQ JSON file, pick a question, and chat. "
                "The prompt template forces a single answer label."
            )
            model_fixed = gr.Textbox(
                label="Model (fixed)",
                value="MedGemma 27B Text",
                interactive=False,
            )
            prompt_template = gr.Textbox(
                label="Prompt template (answer-only)",
                value=MCQ_PROMPT_TEMPLATE,
                lines=9,
                interactive=False,
            )
            mcq_file = gr.File(label="MCQ JSON file", file_types=[".json"])
            load_button = gr.Button("Load questions")
            mcq_state = gr.State([])
            question_choice = gr.Dropdown(label="Question", choices=[], value=None)
            question_preview = gr.Textbox(label="Question preview", lines=6, interactive=False)
            ask_button = gr.Button("Ask selected question")

            chat = gr.Chatbot(label="Chat")
            chat_input = gr.Textbox(label="Message", placeholder="Ask a follow-up or request the answer.")
            send_button = gr.Button("Send", variant="primary")
            clear_chat = gr.Button("Clear chat")

            def _load_questions(file_obj):
                items, options = _load_mcq_file(file_obj)
                return items, gr.update(choices=options, value=options[0][1] if options else None)

            load_button.click(
                _load_questions,
                inputs=[mcq_file],
                outputs=[mcq_state, question_choice],
            )

            def _show_question(items, item_id):
                item = _get_mcq_by_id(items, item_id)
                if not item:
                    return ""
                return _build_mcq_prompt(item)

            question_choice.change(
                _show_question,
                inputs=[mcq_state, question_choice],
                outputs=[question_preview],
            )

            def _send_selected_question_stream(history, items, item_id):
                item = _get_mcq_by_id(items, item_id)
                prompt = _build_mcq_prompt(item)
                if not prompt:
                    yield history, ""
                    return
                yield from _chat_with_model_stream(history, prompt, "MedGemma 27B Text")

            ask_button.click(
                _send_selected_question_stream,
                inputs=[chat, mcq_state, question_choice],
                outputs=[chat, chat_input],
            )

            send_button.click(
                _chat_with_model_stream,
                inputs=[chat, chat_input, model_fixed],
                outputs=[chat, chat_input],
            )
            chat_input.submit(
                _chat_with_model_stream,
                inputs=[chat, chat_input, model_fixed],
                outputs=[chat, chat_input],
            )
            clear_chat.click(lambda: [], outputs=[chat])


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
