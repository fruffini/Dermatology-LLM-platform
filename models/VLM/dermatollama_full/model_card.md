# DermatoLLama Full

Base model: `meta-llama/Llama-3.2-11B-Vision-Instruct`  
Adapter: `DermaVLM/DermatoLLama-full`

This model combines the Llama 3.2 Vision-Instruct backbone with a dermatology-focused LoRA adapter to generate detailed clinical descriptions from images.

Notes:
- Provide a concise clinical prompt to guide the report.
- Longer generations can be slow; adjust max tokens if needed.
