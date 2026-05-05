"""vLLM-Omni TTS Engine for Qwen3-TTS streaming generation."""

from __future__ import annotations

import os
import asyncio
import uuid
import logging
from collections.abc import AsyncGenerator
from typing import Any

import torch

# Must be set before importing vllm_omni
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

from vllm_omni import AsyncOmni
from vllm_omni.inputs.data import OmniPromptType

from .config import (
    SPEAKERS,
    VALID_LANGUAGES,
    MODEL_IDS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_MODEL_TYPE,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MODELS_DIR,
)

logger = logging.getLogger("qwen3_tts_server")


def _extract_sample_rate(raw: object, fallback: int = DEFAULT_SAMPLE_RATE) -> int:
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, list) and raw:
        val = raw[-1]
        if isinstance(val, (int, float)):
            return int(val)
        item_fn = getattr(val, "item", None)
        if callable(item_fn):
            result = item_fn()
            if isinstance(result, (int, float)):
                return int(result)
    item_fn = getattr(raw, "item", None)
    if callable(item_fn):
        result = item_fn()
        if isinstance(result, (int, float)):
            return int(result)
    return fallback


class VLLMEngine:
    """vLLM-Omni based TTS engine with true streaming generation.

    Each model type (custom_voice, voice_design, base) gets its own
    AsyncOmni instance so they can be loaded/unloaded independently.
    """

    def __init__(
        self,
        model_type: str = DEFAULT_MODEL_TYPE,
        model_id: str | None = None,
        models_dir: str = DEFAULT_MODELS_DIR,
        model_manager: Any | None = None,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        device: str = "cuda:0",
        stage_configs_path: str | None = None,
    ):
        self.model_type: str = model_type
        self.model_id: str = model_id or MODEL_IDS.get(model_type, MODEL_IDS[DEFAULT_MODEL_TYPE])
        self.models_dir: str = models_dir
        self.model_manager = model_manager
        self.gpu_memory_utilization: float = gpu_memory_utilization
        self.device: str = device
        self.stage_configs_path: str | None = stage_configs_path
        self.sample_rate: int = DEFAULT_SAMPLE_RATE

        self._omni: AsyncOmni | None = None
        self._loading: bool = False
        self._cancel_events: dict[str, asyncio.Event] = {}

    async def load(self) -> None:
        if self._omni is not None:
            return
        if self._loading:
            return
        self._loading = True
        try:
            model_path = self.model_id

            # Try ModelManager first (supports managed downloads)
            if self.model_manager is not None:
                local = self.model_manager.get_local_path(self.model_type)
                if local is not None:
                    logger.info("Using local model via ModelManager: %s", local)
                    model_path = local
            # Fallback: check local dir directly
            elif os.path.isdir(os.path.join(self.models_dir, self.model_type)):
                local_path = os.path.join(self.models_dir, self.model_type)
                if any(os.scandir(local_path)):
                    logger.info("Using local model at %s", local_path)
                    model_path = local_path

            kwargs: dict[str, Any] = {
                "gpu_memory_utilization": self.gpu_memory_utilization,
            }
            if self.stage_configs_path:
                kwargs["stage_configs_path"] = self.stage_configs_path

            self._omni = AsyncOmni(model=model_path, **kwargs)
        finally:
            self._loading = False

    @property
    def is_loaded(self) -> bool:
        return self._omni is not None

    def _resolve_model_path(self) -> str:
        """Return the local model path if available, otherwise the HF model ID."""
        local_path = os.path.join(self.models_dir, self.model_type)
        if os.path.isdir(local_path) and any(os.scandir(local_path)):
            return local_path
        return self.model_id

    def request_cancel(self, request_id: str) -> bool:
        """Request cancellation of a running generation."""
        if request_id in self._cancel_events:
            self._cancel_events[request_id].set()
            return True
        return False

    async def _estimate_prompt_len(
        self,
        additional_information: dict[str, Any],
    ) -> int:
        """Estimate prompt_token_ids placeholder length for the Talker stage.

        The AR Talker replaces all input embeddings via preprocess, so the
        placeholder values are irrelevant but the length must match the
        embeddings that preprocess will produce.
        """
        try:
            from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
                Qwen3TTSConfig,
            )
            from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
                Qwen3TTSTalkerForConditionalGeneration,
            )
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                self._resolve_model_path(), trust_remote_code=True, padding_side="left"
            )
            cfg = Qwen3TTSConfig.from_pretrained(self._resolve_model_path(), trust_remote_code=True)

            task_type = (additional_information.get("task_type") or ["CustomVoice"])[0]

            return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
                additional_information=additional_information,
                task_type=task_type,
                tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
                codec_language_id=getattr(cfg, "codec_language_id", None),
                spk_is_dialect=getattr(cfg, "spk_is_dialect", None),
                estimate_ref_code_len=lambda _: None,
            )
        except Exception:
            # Fallback: rough estimate
            text = additional_information.get("text", [""])[0]
            extra = additional_information.get("instruct", [""])[0]
            return max(100, int((len(text) + len(extra)) * 1.5) + 256)

    async def generate_stream(
        self,
        text: str,
        language: str = "Chinese",
        speaker: str | None = None,
        instruct: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        x_vector_only_mode: bool = False,
        max_new_tokens: int = 2048,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream TTS generation, yielding audio chunks as they arrive.

        Yields dicts with:
          - type: "audio_chunk" | "audio_done" | "error"
          - audio: numpy array (for audio_chunk)
          - sample_rate: int (for audio_chunk)
          - total_duration: float (for audio_done, in seconds)
          - message: str (for error)
        """
        if not self._omni:
            await self.load()
        assert self._omni is not None

        # Determine task type from model_type
        task_type = self._task_type_from_model()

        # Validate inputs
        if task_type == "CustomVoice":
            speaker = speaker or "Vivian"
            if speaker not in SPEAKERS:
                yield {"type": "error", "message": f"Unknown speaker: {speaker}"}
                return
        if language not in VALID_LANGUAGES:
            yield {"type": "error", "message": f"Unknown language: {language}"}
            return
        if task_type == "VoiceDesign" and not instruct:
            yield {"type": "error", "message": "instruct is required for VoiceDesign"}
            return
        if task_type == "Base" and not ref_audio:
            yield {"type": "error", "message": "ref_audio is required for voice cloning"}
            return

        additional_info: dict[str, Any] = {
            "task_type": [task_type],
            "text": [text],
            "language": [language],
            "max_new_tokens": [max_new_tokens],
        }
        if task_type == "CustomVoice":
            additional_info["speaker"] = [speaker]
            additional_info["instruct"] = [instruct or ""]
        elif task_type == "VoiceDesign":
            additional_info["instruct"] = [instruct]
            additional_info["non_streaming_mode"] = [False]
        elif task_type == "Base":
            additional_info["ref_audio"] = [ref_audio]
            additional_info["ref_text"] = [ref_text or ""]
            additional_info["x_vector_only_mode"] = [x_vector_only_mode]

        prompt_len = await self._estimate_prompt_len(additional_info)
        prompt: OmniPromptType = {  # type: ignore[typeddict-item]
            "prompt_token_ids": [0] * prompt_len,
            "additional_information": additional_info,
        }

        request_id = str(uuid.uuid4())[:8]
        cancel_event = asyncio.Event()
        self._cancel_events[request_id] = cancel_event

        audio_chunks: list[torch.Tensor] = []
        sr = self.sample_rate
        last_mm: dict[str, Any] = {}

        try:
            async for stage_output in self._omni.generate(prompt, request_id=request_id):
                if cancel_event.is_set():
                    yield {"type": "error", "message": "Generation cancelled"}
                    return

                mm = stage_output.multimodal_output
                if not mm:
                    continue

                last_mm = mm
                audio = mm.get("audio")
                if audio is None:
                    continue

                new_chunks: list[torch.Tensor] = []
                if isinstance(audio, list):
                    audio_chunks.extend(audio)
                    new_chunks = audio
                else:
                    audio_chunks.append(audio)
                    new_chunks = [audio]

                incremental = torch.cat(new_chunks, dim=-1)
                audio_np = incremental.float().cpu().numpy().flatten()
                yield {
                    "type": "audio_chunk",
                    "audio": audio_np,
                    "sample_rate": sr,
                }

            if audio_chunks:
                sr = _extract_sample_rate(last_mm.get("sr"), sr)
                audio_tensor = torch.cat(audio_chunks, dim=-1)
                audio_np = audio_tensor.float().cpu().numpy().flatten()
                total_duration = len(audio_np) / sr
                yield {
                    "type": "audio_done",
                    "audio": audio_np,
                    "sample_rate": sr,
                    "total_duration": total_duration,
                }
            else:
                yield {"type": "error", "message": "No audio generated"}
        except asyncio.CancelledError:
            yield {"type": "error", "message": "Generation cancelled"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
        finally:
            self._cancel_events.pop(request_id, None)

    def _task_type_from_model(self) -> str:
        """Determine task type from the loaded model_type."""
        mapping = {
            "custom_voice": "CustomVoice",
            "voice_design": "VoiceDesign",
            "base": "Base",
        }
        return mapping.get(self.model_type, "CustomVoice")

    async def switch_model(self, model_type: str) -> None:
        """Switch to a different model type (unloads current, loads new)."""
        if model_type == self.model_type and self._omni is not None:
            return
        self.unload()
        self.model_type = model_type
        self.model_id = MODEL_IDS.get(model_type, MODEL_IDS[DEFAULT_MODEL_TYPE])
        await self.load()

    def unload(self) -> None:
        """Unload the current model and free GPU memory."""
        if self._omni is not None:
            del self._omni
            self._omni = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_status(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "model_id": self.model_id,
            "models_dir": self.models_dir,
            "is_loaded": self.is_loaded,
            "is_loading": self._loading,
            "sample_rate": self.sample_rate,
            "device": self.device,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
