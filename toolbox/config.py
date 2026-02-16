import os

OUTPUT_DIR = "outputs"
MODEL_ROOT = os.path.join("models", "VLM")
PRIVATE_TOKEN = os.getenv("VLM_PRIVATE_TOKEN", "")
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0").lower() in ("1", "true", "yes")
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "0").lower() in ("1", "true", "yes")
ATTN_IMPLEMENTATION = "flash_attention_2" if USE_FLASH_ATTENTION else "sdpa"

# Options: "4bit", "8bit", "none"
QUANTIZATION_MODE = os.getenv("QUANTIZATION_MODE", "4bit").lower()

VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
USE_VLLM = os.getenv("USE_VLLM", "1").lower() in ("1", "true", "yes")
VLLM_LOG_DIR = os.getenv("VLLM_LOG_DIR", os.path.join(OUTPUT_DIR, "vllm_logs"))
VLLM_WORKDIR = os.getenv("VLLM_WORKDIR", os.getcwd())

VLLM_MODEL_CONFIGS = {
    "MedGemma 27B Text": {
        "model_id": "google/medgemma-27b-it",
        "quantization": "bitsandbytes",
        "load_format": "bitsandbytes",
        "max_model_len": 4096,
        "startup_timeout": 900,
    },
    "MedGemma 4B Text": {
        "model_id": "google/medgemma-4b-it",
        "quantization": None,
        "load_format": "auto",
        "max_model_len": 4096,
        "startup_timeout": 300,
    },
    "Qwen QwQ-32B": {
        "model_id": "Qwen/QwQ-32B",
        "quantization": "bitsandbytes",
        "load_format": "bitsandbytes",
        "max_model_len": 4096,
        "startup_timeout": 1200,
    },
    "II-Medical-8B": {
        "model_id": "Intelligent-Internet/II-Medical-8B",
        "quantization": None,
        "load_format": "auto",
        "max_model_len": 4096,
        "startup_timeout": 300,
    },
}

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
        "model_type": "vllm",
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
