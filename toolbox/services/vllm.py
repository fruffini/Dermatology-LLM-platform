import atexit
import os
import signal
import subprocess
import threading
import time
from datetime import datetime

import requests
from openai import OpenAI

from toolbox.config import (
    USE_VLLM,
    VLLM_API_KEY,
    VLLM_HOST,
    VLLM_LOG_DIR,
    VLLM_MODEL_CONFIGS,
    VLLM_PORT,
    VLLM_WORKDIR,
)


class VLLMServerManager:
    """Manages vLLM server lifecycle - starts/stops servers for different models."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/v1"
        self.process = None
        self.current_model = None
        self.client = None
        self.last_start_error = None
        self.last_log_path = None
        self._log_file = None
        self._lock = threading.RLock()

    def _close_log_file(self):
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def _prepare_log_file(self, model_name: str):
        os.makedirs(VLLM_LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in model_name).strip("_")
        self.last_log_path = os.path.join(VLLM_LOG_DIR, f"{safe_name}_{timestamp}.log")
        self._log_file = open(self.last_log_path, "w", encoding="utf-8", buffering=1)

    def _tail_log(self, lines: int = 20) -> str:
        if not self.last_log_path or not os.path.exists(self.last_log_path):
            return ""
        try:
            with open(self.last_log_path, "r", encoding="utf-8") as fh:
                data = fh.readlines()
            return "".join(data[-lines:]).strip()
        except Exception:
            return ""

    def _kill_existing_vllm(self):
        try:
            subprocess.run(
                f"lsof -ti:{self.port} | xargs -r kill -9",
                shell=True,
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                "pkill -9 -f 'vllm serve'",
                shell=True,
                capture_output=True,
                timeout=10,
            )
            time.sleep(2)
        except Exception:
            pass

    def _wait_for_server(self, timeout: int = 300) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                return_code = self.process.returncode
                log_tail = self._tail_log()
                self.last_start_error = f"vLLM exited with code {return_code}."
                if log_tail:
                    self.last_start_error += f" Last log lines:\n{log_tail}"
                return False
            try:
                response = requests.get(f"http://{self.host}:{self.port}/health", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        log_tail = self._tail_log()
        self.last_start_error = f"vLLM did not become healthy within {timeout}s."
        if log_tail:
            self.last_start_error += f" Last log lines:\n{log_tail}"
        return False

    def start_server(self, model_name: str) -> bool:
        with self._lock:
            self.last_start_error = None
            if self.current_model == model_name and self.is_running():
                return True

            config = VLLM_MODEL_CONFIGS.get(model_name)
            if not config:
                return False

            self.stop_server()

            cmd = [
                "vllm",
                "serve",
                config["model_id"],
                "--dtype",
                "bfloat16",
                "--max-model-len",
                str(config["max_model_len"]),
                "--enforce-eager",
                "--disable-custom-all-reduce",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--gpu-memory-utilization",
                "0.9",
            ]

            if config["quantization"]:
                cmd.extend(["--quantization", config["quantization"]])
            if config["load_format"] != "auto":
                cmd.extend(["--load-format", config["load_format"]])

            env = os.environ.copy()
            env["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
            env["CUDA_VISIBLE_DEVICES"] = "1"
            self._close_log_file()
            self._prepare_log_file(model_name)
            self._log_file.write(
                f"[{datetime.now().isoformat()}] Starting vLLM for {model_name}\n"
                f"Command: {' '.join(cmd)}\n"
                f"Working directory: {VLLM_WORKDIR}\n\n"
            )

            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=self._log_file,
                    stderr=subprocess.STDOUT,
                    cwd=VLLM_WORKDIR,
                    env=env,
                    preexec_fn=os.setsid,
                )
            except Exception as exc:
                self.last_start_error = str(exc)
                self._close_log_file()
                self.process = None
                return False

            startup_timeout = int(config.get("startup_timeout", 300))
            if self._wait_for_server(timeout=startup_timeout):
                self.current_model = model_name
                self.client = OpenAI(base_url=self.base_url, api_key=VLLM_API_KEY)
                return True

            self.stop_server()
            return False

    def stop_server(self, force_orphan_cleanup: bool = False):
        with self._lock:
            if self.process:
                try:
                    child_pgid = os.getpgid(self.process.pid)
                    parent_pgid = os.getpgrp()
                    if child_pgid != parent_pgid:
                        os.killpg(child_pgid, signal.SIGTERM)
                    else:
                        self.process.terminate()
                    self.process.wait(timeout=10)
                except Exception:
                    try:
                        child_pgid = os.getpgid(self.process.pid)
                        parent_pgid = os.getpgrp()
                        if child_pgid != parent_pgid:
                            os.killpg(child_pgid, signal.SIGKILL)
                        else:
                            self.process.kill()
                    except Exception:
                        pass
                self.process = None

            self._close_log_file()
            if force_orphan_cleanup:
                self._kill_existing_vllm()
            self.current_model = None
            self.client = None

    def is_running(self) -> bool:
        if not self.process:
            return False
        try:
            response = requests.get(f"http://{self.host}:{self.port}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def get_client(self):
        return self.client


_VLLM_MANAGER = VLLMServerManager(host=VLLM_HOST, port=VLLM_PORT) if USE_VLLM else None


def get_vllm_manager():
    return _VLLM_MANAGER


def _cleanup_vllm_server():
    if _VLLM_MANAGER:
        try:
            _VLLM_MANAGER.stop_server(force_orphan_cleanup=True)
        except Exception:
            pass


def _handle_shutdown_signal(signum, frame):
    _cleanup_vllm_server()
    raise SystemExit(0)


atexit.register(_cleanup_vllm_server)
signal.signal(signal.SIGINT, _handle_shutdown_signal)
signal.signal(signal.SIGTERM, _handle_shutdown_signal)
