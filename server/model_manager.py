"""Model management: check completeness, download, delete, cancel."""

from __future__ import annotations

import os
import shutil
import threading
import logging
from typing import Any

from .config import (
    MODEL_IDS,
    MODELSCOPE_IDS,
    MODEL_DESCRIPTIONS,
    MODEL_ESSENTIAL_FILES,
    DEFAULT_MODELS_DIR,
)

logger = logging.getLogger("qwen3_tts_server")


class ModelManager:
    def __init__(self, models_dir: str = DEFAULT_MODELS_DIR):
        self.models_dir = models_dir
        self._cancel_events: dict[str, threading.Event] = {}
        self._download_threads: dict[str, threading.Thread] = {}
        self._download_status: dict[str, dict[str, Any]] = {}

    def check_model(self, model_type: str) -> dict[str, Any]:
        local_path = os.path.join(self.models_dir, model_type)
        exists = os.path.isdir(local_path)
        missing: list[str] = []
        present: list[str] = []
        total_size = 0.0
        if exists:
            for f in MODEL_ESSENTIAL_FILES:
                fp = os.path.join(local_path, f)
                if os.path.isfile(fp):
                    present.append(f)
                    total_size += os.path.getsize(fp)
                else:
                    missing.append(f)
        return {
            "model_type": model_type,
            "local_path": local_path,
            "exists": exists,
            "complete": exists and len(missing) == 0,
            "missing_files": missing,
            "present_files": present,
            "total_size_mb": round(total_size / 1048576, 1),
            "description": MODEL_DESCRIPTIONS.get(model_type, ""),
            "model_id": MODEL_IDS.get(model_type, ""),
            "modelscope_id": MODELSCOPE_IDS.get(model_type, ""),
        }

    def check_all(self) -> list[dict[str, Any]]:
        return [self.check_model(mt) for mt in MODEL_IDS]

    def delete_model(self, model_type: str) -> bool:
        local_path = os.path.join(self.models_dir, model_type)
        if os.path.isdir(local_path):
            shutil.rmtree(local_path)
            logger.info("Deleted model: %s", local_path)
            return True
        return False

    def download_model(self, model_type: str, source: str = "modelscope") -> dict[str, str]:
        info = self.check_model(model_type)
        if info["complete"]:
            return {"status": "already_complete", "model_type": model_type}
        if model_type in self._download_threads and self._download_threads[model_type].is_alive():
            return {"status": "already_downloading", "model_type": model_type}

        cancel_event = threading.Event()
        self._cancel_events[model_type] = cancel_event
        self._download_status[model_type] = {
            "status": "downloading",
            "progress": 0,
            "message": "准备下载...",
        }

        t = threading.Thread(
            target=self._download_worker, args=(model_type, cancel_event, source), daemon=True
        )
        self._download_threads[model_type] = t
        t.start()
        return {"status": "started", "model_type": model_type}

    def _download_worker(self, model_type: str, cancel_event: threading.Event, source: str = "modelscope") -> None:
        local_path = os.path.join(self.models_dir, model_type)
        os.makedirs(local_path, exist_ok=True)
        ms_id = MODELSCOPE_IDS.get(model_type, MODEL_IDS.get(model_type, ""))
        hf_id = MODEL_IDS.get(model_type, "")

        try:
            if cancel_event.is_set():
                self._download_status[model_type] = {
                    "status": "cancelled",
                    "progress": 0,
                    "message": "下载已取消",
                }
                shutil.rmtree(local_path, ignore_errors=True)
                return

            # Determine order based on source preference
            try_modelscope = source in ("modelscope", "auto")
            try_huggingface = source in ("huggingface", "auto")

            if try_modelscope:
                self._download_status[model_type] = {
                    "status": "downloading",
                    "progress": 10,
                    "message": f"从 ModelScope 下载: {ms_id}",
                }
                logger.info("Downloading from ModelScope: %s -> %s", ms_id, local_path)

                try:
                    from modelscope.hub.snapshot_download import snapshot_download as ms_download

                    ms_download(model_id=ms_id, local_dir=local_path)
                    if cancel_event.is_set():
                        self._download_status[model_type] = {
                            "status": "cancelled",
                            "progress": 0,
                            "message": "下载已取消",
                        }
                        shutil.rmtree(local_path, ignore_errors=True)
                        return
                    self._download_status[model_type] = {
                        "status": "complete",
                        "progress": 100,
                        "message": "ModelScope 下载完成",
                    }
                    logger.info("ModelScope download complete: %s", local_path)
                    return
                except Exception as ms_exc:
                    logger.info("ModelScope failed: %s", ms_exc)
                    if source != "auto":
                        # User explicitly chose ModelScope, don't fall back
                        self._download_status[model_type] = {
                            "status": "error",
                            "progress": 0,
                            "message": f"ModelScope 下载失败: {ms_exc}",
                        }
                        shutil.rmtree(local_path, ignore_errors=True)
                        return
                    if cancel_event.is_set():
                        self._download_status[model_type] = {
                            "status": "cancelled",
                            "progress": 0,
                            "message": "下载已取消",
                        }
                        shutil.rmtree(local_path, ignore_errors=True)
                        return

            if try_huggingface:
                self._download_status[model_type] = {
                    "status": "downloading",
                    "progress": 30,
                    "message": f"从 HuggingFace 下载: {hf_id}",
                }
                logger.info("Downloading from HuggingFace: %s -> %s", hf_id, local_path)

                try:
                    import huggingface_hub

                    huggingface_hub.snapshot_download(
                        repo_id=hf_id, local_dir=local_path
                    )
                    if cancel_event.is_set():
                        self._download_status[model_type] = {
                            "status": "cancelled",
                            "progress": 0,
                            "message": "下载已取消",
                        }
                        shutil.rmtree(local_path, ignore_errors=True)
                        return
                    self._download_status[model_type] = {
                        "status": "complete",
                        "progress": 100,
                        "message": "HuggingFace 下载完成",
                    }
                    logger.info("HuggingFace download complete: %s", local_path)
                except Exception as hf_exc:
                    logger.exception("HuggingFace download failed: %s", hf_exc)
                    self._download_status[model_type] = {
                        "status": "error",
                        "progress": 0,
                        "message": f"HuggingFace 下载失败: {hf_exc}",
                    }
                    shutil.rmtree(local_path, ignore_errors=True)

            if not try_modelscope and not try_huggingface:
                self._download_status[model_type] = {
                    "status": "error",
                    "progress": 0,
                    "message": f"未知下载源: {source}",
                }
        except Exception as e:
            logger.exception("Download worker error: %s", e)
            self._download_status[model_type] = {
                "status": "error",
                "progress": 0,
                "message": f"下载失败: {e}",
            }

    def cancel_download(self, model_type: str) -> dict[str, str]:
        event = self._cancel_events.get(model_type)
        if event and not event.is_set():
            event.set()
            self._download_status[model_type] = {
                "status": "cancelling",
                "progress": 0,
                "message": "正在取消...",
            }
            return {"status": "cancelling", "model_type": model_type}
        return {"status": "not_downloading", "model_type": model_type}

    def get_download_status(self, model_type: str) -> dict[str, Any]:
        status = self._download_status.get(
            model_type, {"status": "idle", "progress": 0, "message": ""}
        )
        info = self.check_model(model_type)
        return {
            "model_type": model_type,
            "download_status": status["status"],
            "download_progress": status["progress"],
            "download_message": status["message"],
            "model_exists": info["exists"],
            "model_complete": info["complete"],
        }

    def get_local_path(self, model_type: str) -> str | None:
        info = self.check_model(model_type)
        if info["complete"]:
            return info["local_path"]
        return None
