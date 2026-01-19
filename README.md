# Dermatology VLM Inference (Gradio)

Minimal Gradio UI to run dermatology VLM inference on images with a token, and optionally save results as `txt` or `json`.

## Requirements

- Python 3.9+
- `gradio`

Install dependencies:

```bash
pip install gradio
```

## Run the app

```bash
python app.py
```

The UI starts locally and prints a URL in the terminal.

## How to use

1. Select a VLM model from the dropdown.
2. Upload a dermatology image.
3. Enter your token.
4. Choose whether to save the result and pick `txt` or `json`.
5. Click "Run inference".

Saved files are written to `outputs/`.

## Customize VLM inference

Open `app.py` and replace the placeholder logic in `run_inference()` with calls to your real VLMs. Use the `model_name`, `image`, and `token` inputs to route to the correct model.

## Project files

- `app.py` - Gradio UI and inference placeholder
- `.gitignore` - Python/Gradio ignores
- `outputs/` - saved results (created at runtime)
