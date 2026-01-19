import json
import os
from datetime import datetime

import gradio as gr

OUTPUT_DIR = "outputs"


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def _image_summary(image) -> str:
    if image is None:
        return "no image"
    try:
        width, height = image.size
        return f"{width}x{height}"
    except Exception:
        return "unknown"


def run_inference(model_name: str, image, token: str, save_result: bool, save_format: str):
    if image is None:
        return "Please upload an image before running inference.", None

    token = token or ""
    image_info = _image_summary(image)

    # Placeholder response until real VLMs are wired in.
    response = (
        f"Model: {model_name}\n"
        f"Token: {token}\n"
        f"Image: {image_info}\n\n"
        "Result:\n"
        "This is a placeholder response. Wire in your VLM inference here."
    )

    if not save_result:
        return response, None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_model = _safe_slug(model_name) or "vlm"
    filename = f"{timestamp}_{safe_model}.{save_format}"
    path = os.path.join(OUTPUT_DIR, filename)

    if save_format == "json":
        payload = {
            "model": model_name,
            "token": token,
            "image": image_info,
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
            choices=["VLM-A", "VLM-B", "VLM-C"],
            value="VLM-A",
        )
        token_input = gr.Textbox(label="Token", placeholder="Enter access token")

    image_input = gr.Image(label="Dermatology image", type="pil")

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
        inputs=[model_choice, image_input, token_input, save_result, save_format],
        outputs=[output_text, output_file],
    )
    clear_button.click(
        lambda: (None, "", "VLM-A", True, "txt", "", None),
        outputs=[image_input, token_input, model_choice, save_result, save_format, output_text, output_file],
    )


if __name__ == "__main__":
    demo.launch()
