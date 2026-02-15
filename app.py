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
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    TextIteratorStreamer,
    pipeline,
)
from huggingface_hub import InferenceClient
import threading
import subprocess
import time
import signal
import requests
from transformers import StoppingCriteria, StoppingCriteriaList
from openai import OpenAI

# Global stop event for cancelling generation
_STOP_GENERATION = threading.Event()


class StopOnEvent(StoppingCriteria):
    """Custom stopping criteria that checks a threading Event."""
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()


def stop_generation():
    """Signal the model to stop generating."""
    _STOP_GENERATION.set()
    return "Generation stopped."


def reset_stop_flag():
    """Reset the stop flag before starting generation."""
    _STOP_GENERATION.clear()

OUTPUT_DIR = "outputs"
MODEL_ROOT = os.path.join("models", "VLM")
PRIVATE_TOKEN = os.getenv("VLM_PRIVATE_TOKEN", "")
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")


# Use the second GPU (0-based index) when available.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Auto-tune for faster convolutions
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

# Performance optimization flags
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0").lower() in ("1", "true", "yes")
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "0").lower() in ("1", "true", "yes")
ATTN_IMPLEMENTATION = "flash_attention_2" if USE_FLASH_ATTENTION else "sdpa"

# Quantization config for large models (MedGemma 27B)
# Options: "4bit" (fastest, ~14GB VRAM), "8bit" (~27GB VRAM), "none" (full precision, ~54GB VRAM)
QUANTIZATION_MODE = os.getenv("QUANTIZATION_MODE", "4bit").lower()

# vLLM server configuration
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
USE_VLLM = os.getenv("USE_VLLM", "1").lower() in ("1", "true", "yes")

# vLLM model configurations
VLLM_MODEL_CONFIGS = {
    "MedGemma 27B Text": {
        "model_id": "google/medgemma-27b-it",
        "quantization": "bitsandbytes",
        "load_format": "bitsandbytes",
        "max_model_len": 4096,
    },
    "MedGemma 4B Text": {
        "model_id": "google/medgemma-4b-it",
        "quantization": None,
        "load_format": "auto",
        "max_model_len": 4096,
    },
    "Qwen QwQ-32B": {
        "model_id": "Qwen/QwQ-32B",
        "quantization": "bitsandbytes",
        "load_format": "bitsandbytes",
        "max_model_len": 4096,
    },
    "II-Medical-8B": {
        "model_id": "Intelligent-Internet/II-Medical-8B",
        "quantization": None,
        "load_format": "auto",
        "max_model_len": 4096,
    },
}


class VLLMServerManager:
    """Manages vLLM server lifecycle - starts/stops servers for different models."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/v1"
        self.process = None
        self.current_model = None
        self.client = None
        self._lock = threading.Lock()

    def _kill_existing_vllm(self):
        """Kill any existing vLLM processes on the port."""
        try:
            # Kill by port
            subprocess.run(
                f"lsof -ti:{self.port} | xargs -r kill -9",
                shell=True, capture_output=True, timeout=10
            )
            # Also kill by name
            subprocess.run(
                "pkill -9 -f 'vllm serve'",
                shell=True, capture_output=True, timeout=10
            )
            time.sleep(2)
        except Exception:
            pass

    def _wait_for_server(self, timeout: int = 300) -> bool:
        """Wait for vLLM server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://{self.host}:{self.port}/health", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        return False

    def start_server(self, model_name: str) -> bool:
        """Start vLLM server for the specified model."""
        with self._lock:
            if self.current_model == model_name and self.is_running():
                return True

            config = VLLM_MODEL_CONFIGS.get(model_name)
            if not config:
                return False

            # Stop existing server
            self.stop_server()

            # Build vLLM command
            cmd = [
                "vllm", "serve", config["model_id"],
                "--dtype", "bfloat16",
                "--max-model-len", str(config["max_model_len"]),
                "--enforce-eager",
                "--disable-custom-all-reduce",
                "--host", self.host,
                "--port", str(self.port),
                "--gpu-memory-utilization", "0.9",
            ]

            if config["quantization"]:
                cmd.extend(["--quantization", config["quantization"]])
            if config["load_format"] != "auto":
                cmd.extend(["--load-format", config["load_format"]])

            # Start server process
            env = os.environ.copy()
            env["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,
            )

            # Wait for server to be ready
            if self._wait_for_server():
                self.current_model = model_name
                self.client = OpenAI(base_url=self.base_url, api_key=VLLM_API_KEY)
                return True
            else:
                self.stop_server()
                return False

    def stop_server(self):
        """Stop the vLLM server."""
        with self._lock:
            if self.process:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                    self.process.wait(timeout=10)
                except Exception:
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    except Exception:
                        pass
                self.process = None

            self._kill_existing_vllm()
            self.current_model = None
            self.client = None

    def is_running(self) -> bool:
        """Check if vLLM server is running."""
        if not self.process:
            return False
        try:
            response = requests.get(f"http://{self.host}:{self.port}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def get_client(self):
        """Get the OpenAI client for the running server."""
        return self.client


# Global vLLM server manager
_VLLM_MANAGER = VLLMServerManager(host=VLLM_HOST, port=VLLM_PORT) if USE_VLLM else None

def _get_quantization_config():
    """Get BitsAndBytes quantization config based on QUANTIZATION_MODE."""
    if QUANTIZATION_MODE == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif QUANTIZATION_MODE == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None

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
        "model_id": "google/medgemma-27b-it",
        "base_prompt": "Answer the multiple-choice question with only the best answer label.",
        "requires_image_token": False,
        "model_type": "vllm",  # Uses vLLM server for fast inference
    },
    "MedGemma 4B Text": {
        "model_id": "google/medgemma-4b-it",
        "base_prompt": "Answer the multiple-choice question with only the best answer label.",
        "requires_image_token": False,
        "model_type": "vllm",
    },
    "II-Medical-8B": {
        "model_id": "Intelligent-Internet/II-Medical-8B",
        "base_prompt": "Answer the multiple-choice question with only the best answer label.",
        "requires_image_token": False,
        "model_type": "vllm",
    },
    "Qwen QwQ-32B": {
        "model_id": "Qwen/QwQ-32B",
        "base_prompt": "Answer the multiple-choice question with only the best answer label.",
        "requires_image_token": False,
        "model_type": "vllm",
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

HISTORY_TAKING_PROMPT_TEMPLATE = (
    "You are an experienced physician conducting a diagnostic interview.\n"
    "Based on the patient's description of their symptoms, provide:\n"
    "1. The most likely diagnosis\n"
    "2. Key clinical features that support this diagnosis\n"
    "3. Important differential diagnoses to consider\n"
    "4. Any red flags or warning signs mentioned\n\n"
    "Patient's description:\n\"{symptoms}\"\n\n"
    "Analysis:"
)
HISTORY_TAKING_MAX_NEW_TOKENS = int(os.getenv("HISTORY_TAKING_MAX_NEW_TOKENS", "512"))


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


def _maybe_compile(model):
    """Apply torch.compile if enabled via USE_TORCH_COMPILE env var."""
    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        return torch.compile(model, mode="reduce-overhead")
    return model


def _load_model(model_name: str):
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
        model = _maybe_compile(model)
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
        model = _maybe_compile(model)
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
            "low_cpu_mem_usage": True,  # Faster loading
        }
        if quant_config:
            # Quantized models: don't use custom attention implementation
            model_kwargs["quantization_config"] = quant_config
        else:
            # Full precision: use optimized attention and dtype
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.eval()
        # Note: torch.compile may not work well with quantized models
        if not quant_config:
            model = _maybe_compile(model)
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
        model = _maybe_compile(model)

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


def _load_history_taking_file(file_obj):
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


def _get_history_case_by_id(items, item_id):
    for item in items:
        if item.get("id") == item_id:
            return item
    return None


def _build_history_taking_prompt(item):
    if not item:
        return ""
    symptoms = item.get("question", "")
    return HISTORY_TAKING_PROMPT_TEMPLATE.format(symptoms=symptoms)


def _generate_medgemma_text(tokenizer, model, message):
    messages = [{"role": "user", "content": message}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    input_length = input_ids.shape[-1]
    # Ensure pad_token_id is set
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
    messages = [{"role": "user", "content": message}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # Ensure pad_token_id is set
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    # Custom stopping criteria for cancellation
    stop_criteria = StoppingCriteriaList([StopOnEvent(_STOP_GENERATION)])
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
        if _STOP_GENERATION.is_set():
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
    import time
    reset_stop_flag()
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
                if _STOP_GENERATION.is_set():
                    break
                if chunk.choices and chunk.choices[0].delta.content:
                    partial += chunk.choices[0].delta.content
                    yield partial.strip()
            return  # Success, exit the retry loop
        except Exception as e:
            error_msg = str(e)
            if "502" in error_msg or "503" in error_msg or "504" in error_msg:
                if attempt < max_retries - 1:
                    yield f"⏳ Server busy, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})"
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    yield f"❌ Server error after {max_retries} attempts: {error_msg}\n\nThe HuggingFace Inference API for this model is currently unavailable. Try again later or use a different model."
            else:
                yield f"❌ Error: {error_msg}"
                return


def _stream_vllm(model_name: str, message: str, max_tokens: int = 256):
    """Stream generation using vLLM server with automatic server management."""
    if not _VLLM_MANAGER:
        yield "❌ vLLM is disabled. Set USE_VLLM=1 to enable."
        return

    reset_stop_flag()

    # Check if we need to start/switch the server
    if _VLLM_MANAGER.current_model != model_name:
        yield f"🔄 Starting vLLM server for {model_name}... (this may take a few minutes)"

        if not _VLLM_MANAGER.start_server(model_name):
            yield f"❌ Failed to start vLLM server for {model_name}. Check logs for details."
            return

        yield f"✅ vLLM server ready for {model_name}"

    client = _VLLM_MANAGER.get_client()
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
            if _STOP_GENERATION.is_set():
                break
            if chunk.choices and chunk.choices[0].delta.content:
                partial += chunk.choices[0].delta.content
                yield partial.strip()
    except Exception as e:
        yield f"❌ vLLM error: {str(e)}"


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


def _chat_with_model_stream(history, message, model_name, max_tokens=None):
    if not message:
        yield history, ""
        return
    history = history or []
    model_type = MODEL_REGISTRY[model_name]["model_type"]
    tokens = max_tokens if max_tokens else 256

    # Use vLLM for supported models (automatic server management)
    if model_type == "vllm" and _VLLM_MANAGER:
        for partial in _stream_vllm(model_name, message, tokens):
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": partial},
            ], ""
        return

    # Fallback to local inference or inference client
    _unload_other_models(model_name)
    processor, model = _load_model(model_name)

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

        with gr.Tab("MCQ Chat"):
            gr.Markdown(
                "Load a formatted MCQ JSON file, pick a question, and chat. "
                "The prompt template forces a single answer label."
            )
            mcq_model_choice = gr.Dropdown(
                label="Model",
                choices=["MedGemma 4B Text", "MedGemma 27B Text", "II-Medical-8B", "Qwen QwQ-32B"],
                value="MedGemma 4B Text",
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
            with gr.Row():
                send_button = gr.Button("Send", variant="primary")
                stop_button = gr.Button("Stop", variant="stop")
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

            def _send_selected_question_stream(history, items, item_id, model_name):
                item = _get_mcq_by_id(items, item_id)
                prompt = _build_mcq_prompt(item)
                if not prompt:
                    yield history, ""
                    return
                yield from _chat_with_model_stream(history, prompt, model_name)

            ask_button.click(
                _send_selected_question_stream,
                inputs=[chat, mcq_state, question_choice, mcq_model_choice],
                outputs=[chat, chat_input],
            )

            send_button.click(
                _chat_with_model_stream,
                inputs=[chat, chat_input, mcq_model_choice],
                outputs=[chat, chat_input],
            )
            stop_button.click(
                stop_generation,
                outputs=[],
            )
            chat_input.submit(
                _chat_with_model_stream,
                inputs=[chat, chat_input, mcq_model_choice],
                outputs=[chat, chat_input],
            )
            clear_chat.click(lambda: [], outputs=[chat])

        with gr.Tab("History Taking Cases"):
            gr.Markdown(
                "Load a history-taking cases JSON file, pick a case, and get diagnostic analysis. "
                "The model will analyze the patient's symptoms and provide a differential diagnosis."
            )
            ht_model_choice = gr.Dropdown(
                label="Model",
                choices=["MedGemma 4B Text", "MedGemma 27B Text", "II-Medical-8B", "Qwen QwQ-32B"],
                value="MedGemma 4B Text",
            )
            ht_prompt_template = gr.Textbox(
                label="Prompt template (diagnostic reasoning)",
                value=HISTORY_TAKING_PROMPT_TEMPLATE,
                lines=12,
                interactive=False,
            )
            ht_file = gr.File(label="History Taking JSON file", file_types=[".json"])
            ht_load_button = gr.Button("Load cases")
            ht_state = gr.State([])
            ht_case_choice = gr.Dropdown(label="Case", choices=[], value=None)
            ht_case_preview = gr.Textbox(label="Patient description", lines=4, interactive=False)
            ht_ask_button = gr.Button("Analyze symptoms")

            ht_chat = gr.Chatbot(label="Diagnostic Analysis")
            ht_chat_input = gr.Textbox(label="Follow-up question", placeholder="Ask follow-up questions about the diagnosis.")
            with gr.Row():
                ht_send_button = gr.Button("Send", variant="primary")
                ht_stop_button = gr.Button("Stop", variant="stop")
                ht_clear_chat = gr.Button("Clear chat")

            def _load_ht_cases(file_obj):
                items, options = _load_history_taking_file(file_obj)
                return items, gr.update(choices=options, value=options[0][1] if options else None)

            ht_load_button.click(
                _load_ht_cases,
                inputs=[ht_file],
                outputs=[ht_state, ht_case_choice],
            )

            def _show_ht_case(items, item_id):
                item = _get_history_case_by_id(items, item_id)
                if not item:
                    return ""
                return item.get("question", "")

            ht_case_choice.change(
                _show_ht_case,
                inputs=[ht_state, ht_case_choice],
                outputs=[ht_case_preview],
            )

            def _send_ht_case_stream(history, items, item_id, model_name):
                item = _get_history_case_by_id(items, item_id)
                prompt = _build_history_taking_prompt(item)
                if not prompt:
                    yield history, ""
                    return
                yield from _chat_with_model_stream(history, prompt, model_name, HISTORY_TAKING_MAX_NEW_TOKENS)

            ht_ask_button.click(
                _send_ht_case_stream,
                inputs=[ht_chat, ht_state, ht_case_choice, ht_model_choice],
                outputs=[ht_chat, ht_chat_input],
            )

            def _ht_chat_stream(history, message, model_name):
                yield from _chat_with_model_stream(history, message, model_name, HISTORY_TAKING_MAX_NEW_TOKENS)

            ht_send_button.click(
                _ht_chat_stream,
                inputs=[ht_chat, ht_chat_input, ht_model_choice],
                outputs=[ht_chat, ht_chat_input],
            )
            ht_stop_button.click(
                stop_generation,
                outputs=[],
            )
            ht_chat_input.submit(
                _ht_chat_stream,
                inputs=[ht_chat, ht_chat_input, ht_model_choice],
                outputs=[ht_chat, ht_chat_input],
            )
            ht_clear_chat.click(lambda: [], outputs=[ht_chat])


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
