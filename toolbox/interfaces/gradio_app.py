import gradio as gr

from toolbox.config import (
    HISTORY_TAKING_MAX_NEW_TOKENS,
    HISTORY_TAKING_PROMPT_TEMPLATE,
    MCQ_MAX_NEW_TOKENS,
    MCQ_PROMPT_TEMPLATE,
    MODEL_REGISTRY,
)
from toolbox.services.core import (
    build_history_taking_prompt,
    build_mcq_prompt,
    chat_with_model_stream,
    get_history_case_by_id,
    get_mcq_by_id,
    load_history_taking_file,
    load_mcq_file,
    load_model_card,
    run_batch_questions,
    run_inference,
    unload_other_models,
)
from toolbox.services.runtime import stop_generation


def build_demo() -> gr.Blocks:
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
                model_card = gr.Markdown(load_model_card("PaliGemma Derm"))

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
                    unload_other_models(name)
                    return (
                        MODEL_REGISTRY[name]["base_prompt"],
                        f"**Suggested prompt**: {MODEL_REGISTRY[name]['base_prompt']}",
                        load_model_card(name),
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
                    items, options = load_mcq_file(file_obj)
                    return items, gr.update(choices=options, value=options[0][1] if options else None)

                load_button.click(
                    _load_questions,
                    inputs=[mcq_file],
                    outputs=[mcq_state, question_choice],
                )

                def _show_question(items, item_id):
                    item = get_mcq_by_id(items, item_id)
                    if not item:
                        return ""
                    return build_mcq_prompt(item)

                question_choice.change(
                    _show_question,
                    inputs=[mcq_state, question_choice],
                    outputs=[question_preview],
                )

                def _send_selected_question_stream(history, items, item_id, model_name):
                    item = get_mcq_by_id(items, item_id)
                    prompt = build_mcq_prompt(item)
                    if not prompt:
                        yield history, ""
                        return
                    yield from chat_with_model_stream(history, prompt, model_name)

                ask_button.click(
                    _send_selected_question_stream,
                    inputs=[chat, mcq_state, question_choice, mcq_model_choice],
                    outputs=[chat, chat_input],
                )

                send_button.click(
                    chat_with_model_stream,
                    inputs=[chat, chat_input, mcq_model_choice],
                    outputs=[chat, chat_input],
                )
                stop_button.click(stop_generation, outputs=[])
                chat_input.submit(
                    chat_with_model_stream,
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
                ht_chat_input = gr.Textbox(
                    label="Follow-up question",
                    placeholder="Ask follow-up questions about the diagnosis.",
                )
                with gr.Row():
                    ht_send_button = gr.Button("Send", variant="primary")
                    ht_stop_button = gr.Button("Stop", variant="stop")
                    ht_clear_chat = gr.Button("Clear chat")

                def _load_ht_cases(file_obj):
                    items, options = load_history_taking_file(file_obj)
                    return items, gr.update(choices=options, value=options[0][1] if options else None)

                ht_load_button.click(
                    _load_ht_cases,
                    inputs=[ht_file],
                    outputs=[ht_state, ht_case_choice],
                )

                def _show_ht_case(items, item_id):
                    item = get_history_case_by_id(items, item_id)
                    if not item:
                        return ""
                    return item.get("question", "")

                ht_case_choice.change(
                    _show_ht_case,
                    inputs=[ht_state, ht_case_choice],
                    outputs=[ht_case_preview],
                )

                def _send_ht_case_stream(history, items, item_id, model_name):
                    item = get_history_case_by_id(items, item_id)
                    prompt = build_history_taking_prompt(item)
                    if not prompt:
                        yield history, ""
                        return
                    yield from chat_with_model_stream(history, prompt, model_name, HISTORY_TAKING_MAX_NEW_TOKENS)

                ht_ask_button.click(
                    _send_ht_case_stream,
                    inputs=[ht_chat, ht_state, ht_case_choice, ht_model_choice],
                    outputs=[ht_chat, ht_chat_input],
                )

                def _ht_chat_stream(history, message, model_name):
                    yield from chat_with_model_stream(history, message, model_name, HISTORY_TAKING_MAX_NEW_TOKENS)

                ht_send_button.click(
                    _ht_chat_stream,
                    inputs=[ht_chat, ht_chat_input, ht_model_choice],
                    outputs=[ht_chat, ht_chat_input],
                )
                ht_stop_button.click(stop_generation, outputs=[])
                ht_chat_input.submit(
                    _ht_chat_stream,
                    inputs=[ht_chat, ht_chat_input, ht_model_choice],
                    outputs=[ht_chat, ht_chat_input],
                )
                ht_clear_chat.click(lambda: [], outputs=[ht_chat])

            with gr.Tab("Batch Questions"):
                gr.Markdown("Run all questions from one file with the selected model and export a JSON report.")
                batch_model_choice = gr.Dropdown(
                    label="Model",
                    choices=["MedGemma 4B Text", "MedGemma 27B Text", "II-Medical-8B", "Qwen QwQ-32B"],
                    value="MedGemma 4B Text",
                )
                batch_file = gr.File(label="Questions file (.json/.txt/.md)", file_types=[".json", ".txt", ".md"])
                with gr.Row():
                    batch_prompt_mode = gr.Radio(
                        label="Prompt mode",
                        choices=["Auto", "MCQ template", "History template", "Raw question"],
                        value="Auto",
                    )
                    batch_max_tokens = gr.Slider(
                        label="Max new tokens",
                        minimum=8,
                        maximum=1024,
                        value=MCQ_MAX_NEW_TOKENS,
                        step=8,
                    )
                batch_run_button = gr.Button("Run all questions", variant="primary")
                batch_status = gr.Textbox(label="Batch status", lines=8)
                batch_output_file = gr.File(label="Batch JSON output")

                batch_run_button.click(
                    run_batch_questions,
                    inputs=[batch_model_choice, batch_file, batch_prompt_mode, batch_max_tokens],
                    outputs=[batch_status, batch_output_file],
                )

    return demo
