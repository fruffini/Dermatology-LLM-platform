import threading

import torch
from transformers import StoppingCriteria

from toolbox.config import USE_TORCH_COMPILE

_STOP_GENERATION = threading.Event()


class StopOnEvent(StoppingCriteria):
    """Custom stopping criteria that checks a threading Event."""

    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()


def configure_torch_runtime() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def maybe_compile(model):
    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        return torch.compile(model, mode="reduce-overhead")
    return model


def stop_generation():
    _STOP_GENERATION.set()
    return "Generation stopped."


def reset_stop_flag():
    _STOP_GENERATION.clear()


def get_stop_event() -> threading.Event:
    return _STOP_GENERATION
