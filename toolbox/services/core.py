import gc
import json
import os
import threading
import time
from datetime import datetime

from huggingface_hub import InferenceClient
from peft import PeftModel
from PIL import Image
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    StoppingCriteriaList,
    TextIteratorStreamer,
    pipeline,
)

from toolbox.config import (
    ATTN_IMPLEMENTATION,
    HF_TOKEN,
    HISTORY_TAKING_MAX_NEW_TOKENS,
    HISTORY_TAKING_PROMPT_TEMPLATE,
    MCQ_MAX_NEW_TOKENS,
    MCQ_PROMPT_TEMPLATE,
    MODEL_REGISTRY,
    OUTPUT_DIR,
    QUANTIZATION_MODE,
    VLLM_MODEL_CONFIGS,
)
from toolbox.services.runtime import (
    StopOnEvent,
    configure_torch_runtime,
    get_stop_event,
    maybe_compile,
    reset_stop_flag,
)
from toolbox.services.vllm import get_vllm_manager

configure_torch_runtime()

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


def _format_choices(choices):
    lines = []
    for choice in choices or []:
        label = choice.get("label", "")
        text = choice.get("text", "")
        lines.append(f"{label}. {text}".strip())
    return "\n".join(lines)


def _get_quantization_config():
    if QUANTIZATION_MODE == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if QUANTIZATION_MODE == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    info = MODEL_REGISTRY[model_name]
    model_type = info["model_type"]

    if model_type == "paligemma":
        model_id = info["model_id"]
        processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation=ATTN_IMPLEMENTATION,
            token=HF_TOKEN,
        )
        model.eval()
        model = maybe_compile(model)
    elif model_type == "medgemma_it":
        model_id = info["model_id"]
        processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=ATTN_IMPLEMENTATION,
            token=HF_TOKEN,
        )
        model.eval()
        model = maybe_compile(model)
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
        quant_config = _get_quantization_config()
        model_kwargs = {
            "device_map": "auto",
            "token": HF_TOKEN,
            "low_cpu_mem_usage": True,
        }
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.eval()
        if not quant_config:
            model = maybe_compile(model)
        _MODEL_CACHE[model_name] = (tokenizer, model)
        return tokenizer, model
    elif model_type == "inference_client":
        client = InferenceClient(api_key=HF_TOKEN)
        _MODEL_CACHE[model_name] = (client, info["model_id"])
        return client, info["model_id"]
    else:
        if not HF_TOKEN:
            raise RuntimeError(
                "Missing Hugging Face token. Set HUGGINGFACE_HUB_TOKEN (or HF_TOKEN) to access gated models."
            )
        base_model_name = info["model_id"]
        processor = AutoProcessor.from_pretrained(base_model_name, token=HF_TOKEN)
        model = MllamaForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=ATTN_IMPLEMENTATION,
            token=HF_TOKEN,
        )
        model = PeftModel.from_pretrained(model, info["adapter_id"], token=HF_TOKEN)
        model.eval()
        model = maybe_compile(model)

    _MODEL_CACHE[model_name] = (processor, model)
    return processor, model


def load_model_card(model_name: str) -> str:
    info = MODEL_REGISTRY.get(model_name, {})
    card_path = info.get("card_path")
    if card_path and os.path.exists(card_path):
        with open(card_path, "r", encoding="utf-8") as handle:
            return handle.read()
    return "No model card available."


def unload_other_models(selected_name: str) -> None:
    to_delete = [name for name in _MODEL_CACHE if name != selected_name]
    for name in to_delete:
        processor, model = _MODEL_CACHE.pop(name)
        del processor
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def run_inference(model_name: str, image_path: str, prompt: str, save_result: bool, save_format: str):
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

    unload_other_models(model_name)
    processor, model = load_model(model_name)
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
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=80, use_cache=True)
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
                use_cache=True,
            )
        generated_tokens = outputs[0][input_length:]
        decoded_output = processor.decode(generated_tokens, skip_special_tokens=True)
    elif model_type == "text_pipeline":
        message_text = prompt or "Explain the finding clearly and concisely."
        messages = [{"role": "user", "content": message_text}]
        outputs = processor(messages, max_new_tokens=256)
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
            "use_cache": True,
        }
        input_length = inputs.input_ids.shape[1]
        with torch.inference_mode():
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


def load_mcq_file(file_obj):
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


def get_mcq_by_id(items, item_id):
    for item in items:
        if item.get("id") == item_id:
            return item
    return None


def build_mcq_prompt(item):
    if not item:
        return ""
    choices = _format_choices(item.get("choices", []))
    return MCQ_PROMPT_TEMPLATE.format(question=item.get("question", ""), choices=choices)


def load_history_taking_file(file_obj):
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


def get_history_case_by_id(items, item_id):
    for item in items:
        if item.get("id") == item_id:
            return item
    return None


def build_history_taking_prompt(item):
    if not item:
        return ""
    symptoms = item.get("question", "")
    return HISTORY_TAKING_PROMPT_TEMPLATE.format(symptoms=symptoms)


def _load_batch_questions(file_obj):
    if not file_obj:
        return [], None
    path = getattr(file_obj, "name", None) or file_obj
    if not path:
        return [], None

    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        items = []
        with open(path, "r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                question = line.strip()
                if question:
                    items.append({"id": f"q{idx}", "question": question, "choices": []})
        return items, path

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        raw_items = data.get("items") or data.get("questions") or data.get("data") or []
    elif isinstance(data, list):
        raw_items = data
    else:
        raw_items = []

    items = []
    for idx, raw in enumerate(raw_items, start=1):
        if isinstance(raw, str):
            items.append({"id": f"q{idx}", "question": raw, "choices": []})
            continue
        if not isinstance(raw, dict):
            continue
        item_id = raw.get("id", f"q{idx}")
        question = (raw.get("question") or raw.get("prompt") or raw.get("input") or "").strip()
        if not question:
            continue
        choices = raw.get("choices", [])
        items.append({"id": item_id, "question": question, "choices": choices})
    return items, path


def _build_batch_prompt(item, prompt_mode: str):
    if not item:
        return ""
    if prompt_mode == "MCQ template":
        choices = item.get("choices", [])
        if choices:
            return build_mcq_prompt(item)
        return item.get("question", "")
    if prompt_mode == "History template":
        return build_history_taking_prompt(item)
    if prompt_mode == "Raw question":
        return item.get("question", "")

    if item.get("choices"):
        return build_mcq_prompt(item)
    return item.get("question", "")


def _infer_batch_data_type(items, prompt_mode: str, source_path: str):
    if prompt_mode == "MCQ template":
        return "mcq"
    if prompt_mode == "History template":
        return "history_taking"
    if prompt_mode == "Raw question":
        return "raw"

    source_name = os.path.basename(source_path or "").lower()
    if "mcq" in source_name:
        return "mcq"
    if "history" in source_name:
        return "history_taking"

    with_choices = sum(1 for item in items if item.get("choices"))
    if with_choices > 0:
        return "mcq"
    return "generic"


def _generate_medgemma_text(tokenizer, model, message):
    messages = [{"role": "user", "content": message}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
        model.device
    )
    input_length = input_ids.shape[-1]
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=MCQ_MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def _stream_medgemma_text(tokenizer, model, message, max_tokens=None):
    if max_tokens is None:
        max_tokens = MCQ_MAX_NEW_TOKENS
    reset_stop_flag()
    stop_event = get_stop_event()
    messages = [{"role": "user", "content": message}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
        model.device
    )
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    stop_criteria = StoppingCriteriaList([StopOnEvent(stop_event)])
    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_tokens,
        "do_sample": False,
        "use_cache": True,
        "streamer": streamer,
        "pad_token_id": pad_token_id,
        "stopping_criteria": stop_criteria,
    }
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    partial = ""
    for token_text in streamer:
        if stop_event.is_set():
            break
        partial += token_text
        yield partial.strip()
    thread.join()


def _generate_inference_client(client, model_id, message, max_tokens=256):
    completion = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": message}],
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


def _stream_inference_client(client, model_id, message, max_tokens=256):
    reset_stop_flag()
    stop_event = get_stop_event()
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            stream = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": message}],
                max_tokens=max_tokens,
                stream=True,
            )
            partial = ""
            for chunk in stream:
                if stop_event.is_set():
                    break
                if chunk.choices and chunk.choices[0].delta.content:
                    partial += chunk.choices[0].delta.content
                    yield partial.strip()
            return
        except Exception as e:
            error_msg = str(e)
            if "502" in error_msg or "503" in error_msg or "504" in error_msg:
                if attempt < max_retries - 1:
                    yield f"⏳ Server busy, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})"
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    yield (
                        f"❌ Server error after {max_retries} attempts: {error_msg}\n\n"
                        "The HuggingFace Inference API for this model is currently unavailable. "
                        "Try again later or use a different model."
                    )
            else:
                yield f"❌ Error: {error_msg}"
                return


def _stream_vllm(model_name: str, message: str, max_tokens: int = 256):
    manager = get_vllm_manager()
    if not manager:
        yield "❌ vLLM is disabled. Set USE_VLLM=1 to enable."
        return

    reset_stop_flag()
    stop_event = get_stop_event()

    if manager.current_model != model_name:
        yield f"🔄 Starting vLLM server for {model_name}... (this may take a few minutes)"

        if not manager.start_server(model_name):
            error_details = manager.last_start_error or "No startup error captured."
            log_path = manager.last_log_path or "(log file not created)"
            yield (
                f"❌ Failed to start vLLM server for {model_name}.\n"
                f"Log file: {log_path}\n"
                f"Details: {error_details}"
            )
            return

        yield f"✅ vLLM server ready for {model_name}"

    client = manager.get_client()
    if not client:
        yield "❌ vLLM client not available."
        return

    model_id = VLLM_MODEL_CONFIGS.get(model_name, {}).get("model_id", model_name)

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": message}],
            max_tokens=max_tokens,
            stream=True,
            temperature=0,
        )
        partial = ""
        for chunk in stream:
            if stop_event.is_set():
                break
            if chunk.choices and chunk.choices[0].delta.content:
                partial += chunk.choices[0].delta.content
                yield partial.strip()
    except Exception as e:
        yield f"❌ vLLM error: {str(e)}"


def _chat_with_model(history, message, model_name):
    if not message:
        return history, ""
    unload_other_models(model_name)
    processor, model = load_model(model_name)
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

    if model_type != "medgemma_text_it":
        generated_text = outputs[0].get("generated_text", "")
        if isinstance(generated_text, list):
            last_item = generated_text[-1] if generated_text else ""
            if isinstance(last_item, dict):
                reply = last_item.get("content", str(last_item))
            else:
                reply = str(last_item)
        else:
            reply = str(generated_text)
            if model_type == "text_pipeline" and reply.startswith(prompt):
                reply = reply[len(prompt) :].strip()

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


def chat_with_model_stream(history, message, model_name, max_tokens=None):
    if not message:
        yield history, ""
        return
    history = history or []
    model_type = MODEL_REGISTRY[model_name]["model_type"]
    tokens = max_tokens if max_tokens else 256

    if model_type == "vllm" and get_vllm_manager():
        for partial in _stream_vllm(model_name, message, tokens):
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": partial},
            ], ""
        return

    unload_other_models(model_name)
    processor, model = load_model(model_name)

    if model_type == "medgemma_text_it":
        for partial in _stream_medgemma_text(processor, model, message, max_tokens):
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": partial},
            ], ""
        return

    if model_type == "inference_client":
        for partial in _stream_inference_client(processor, model, message, tokens):
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": partial},
            ], ""
        return

    new_history, _ = _chat_with_model(history, message, model_name)
    yield new_history, ""


def _generate_text_once(model_name: str, message: str, max_tokens: int = 256):
    model_type = MODEL_REGISTRY[model_name]["model_type"]
    tokens = max_tokens or 256

    if model_type == "vllm" and get_vllm_manager():
        last = ""
        for partial in _stream_vllm(model_name, message, tokens):
            last = partial
        return last

    unload_other_models(model_name)
    processor, model = load_model(model_name)

    if model_type == "medgemma_text_it":
        return _generate_medgemma_text(processor, model, message)

    if model_type == "inference_client":
        return _generate_inference_client(processor, model, message, tokens)

    history, _ = _chat_with_model([], message, model_name)
    if not history:
        return ""
    return history[-1].get("content", "")


def run_batch_questions(model_name: str, file_obj, prompt_mode: str, max_tokens: int):
    max_tokens = int(max_tokens)
    items, source_path = _load_batch_questions(file_obj)
    if not items:
        return "No valid questions found in the selected file.", None
    data_type = _infer_batch_data_type(items, prompt_mode, source_path)

    output_folder = os.path.join(OUTPUT_DIR, "batch_runs")
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_model = _safe_slug(model_name) or "model"
    output_path = os.path.join(output_folder, f"{timestamp}_{safe_model}_{data_type}_batch.json")

    results = []
    for idx, item in enumerate(items, start=1):
        prompt = _build_batch_prompt(item, prompt_mode)
        if not prompt:
            results.append(
                {
                    "index": idx,
                    "id": item.get("id", f"q{idx}"),
                    "patient_comment": item.get("question", ""),
                    "response": "",
                    "error": "Empty prompt after formatting.",
                }
            )
            continue

        try:
            response = _generate_text_once(model_name, prompt, max_tokens)
            error = response if isinstance(response, str) and response.startswith("❌") else ""
        except Exception as exc:
            response = ""
            error = str(exc)

        results.append(
            {
                "index": idx,
                "id": item.get("id", f"q{idx}"),
                "patient_comment": item.get("question", ""),
                "response": response,
                "error": error,
            }
        )

    errors = sum(1 for row in results if row.get("error"))
    payload = {
        "model": model_name,
        "model_id": MODEL_REGISTRY.get(model_name, {}).get("model_id", ""),
        "data_type": data_type,
        "prompt_mode": prompt_mode,
        "max_tokens": max_tokens,
        "source_file": source_path,
        "created_at_utc": timestamp,
        "total_questions": len(results),
        "errors": errors,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    summary = (
        f"Batch completed.\n"
        f"Model: {model_name}\n"
        f"Data type: {data_type}\n"
        f"Questions: {len(results)}\n"
        f"Errors: {errors}\n"
        f"Saved to: {output_path}"
    )
    return summary, output_path


__all__ = [
    "HISTORY_TAKING_MAX_NEW_TOKENS",
    "build_history_taking_prompt",
    "build_mcq_prompt",
    "chat_with_model_stream",
    "get_history_case_by_id",
    "get_mcq_by_id",
    "load_history_taking_file",
    "load_mcq_file",
    "load_model_card",
    "run_batch_questions",
    "run_inference",
    "unload_other_models",
]
