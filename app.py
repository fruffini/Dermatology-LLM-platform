import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from toolbox.interfaces.gradio_app import build_demo


def main() -> None:
    demo = build_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    main()
