"""FastAPI server for Qwen3-TTS streaming generation.

Provides:
  - WebSocket /ws/tts for streaming audio with cancel support
  - POST /v1/audio/speech for REST (returns full WAV)
  - POST /v1/audio/speech/stream for REST streaming (SSE with PCM chunks)
  - POST /v1/audio/speech/cancel to cancel in-progress generation
  - GET /v1/audio/speakers, /v1/audio/languages, /v1/audio/status for metadata
"""

import os
import asyncio
import io
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .config import (
    SPEAKERS,
    LANGUAGES,
    VALID_LANGUAGES,
    MODEL_IDS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_MODEL_TYPE,
    DEFAULT_HOST,
    DEFAULT_PORT,
)
from .vllm_engine import VLLMEngine

logger = logging.getLogger("qwen3_tts_server")

app = FastAPI(title="Qwen3-TTS Server", version="1.0.0")

engine: Optional[VLLMEngine] = None

active_requests: dict[str, WebSocket] = {}


class TTSRequest(BaseModel):
    text: str
    language: str = "Chinese"
    speaker: Optional[str] = "Vivian"
    instruct: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    x_vector_only_mode: bool = False
    max_new_tokens: int = 2048
    model_type: str = DEFAULT_MODEL_TYPE


class CancelRequest(BaseModel):
    request_id: str


def get_engine() -> VLLMEngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine


@app.on_event("startup")
async def startup():
    global engine
    model_type = os.environ.get("QWEN3_TTS_MODEL_TYPE", DEFAULT_MODEL_TYPE)
    gpu_util = float(os.environ.get("QWEN3_TTS_GPU_UTIL", "0.3"))
    device = os.environ.get("QWEN3_TTS_DEVICE", "cuda:0")
    stage_configs = os.environ.get("QWEN3_TTS_STAGE_CONFIGS", None)

    engine = VLLMEngine(
        model_type=model_type,
        gpu_memory_utilization=gpu_util,
        device=device,
        stage_configs_path=stage_configs if stage_configs else None,
    )
    logger.info(f"Engine created: model_type={model_type}, device={device}")


@app.get("/v1/audio/status")
async def status():
    return get_engine().get_status()


@app.get("/v1/audio/speakers")
async def speakers():
    return [
        {"name": name, "description_zh": info["zh"], "description_en": info["en"], "language": info["language"]}
        for name, info in SPEAKERS.items()
    ]


@app.get("/v1/audio/languages")
async def languages():
    return VALID_LANGUAGES


@app.get("/v1/audio/models")
async def models():
    return MODEL_IDS


@app.post("/v1/audio/speech")
async def speech(req: TTSRequest):
    """Non-streaming REST endpoint — returns complete WAV file."""
    eng = get_engine()

    if req.model_type != eng.model_type:
        await eng.switch_model(req.model_type)

    audio_data = None
    sample_rate = DEFAULT_SAMPLE_RATE

    async for chunk in eng.generate_stream(
        text=req.text,
        language=req.language,
        speaker=req.speaker,
        instruct=req.instruct,
        ref_audio=req.ref_audio,
        ref_text=req.ref_text,
        x_vector_only_mode=req.x_vector_only_mode,
        max_new_tokens=req.max_new_tokens,
    ):
        if chunk["type"] == "error":
            raise HTTPException(status_code=500, detail=chunk["message"])
        if chunk["type"] in ("audio_chunk", "audio_done"):
            audio_data = chunk["audio"]
            sample_rate = chunk["sample_rate"]
        if chunk["type"] == "audio_done":
            break

    if audio_data is None:
        raise HTTPException(status_code=500, detail="No audio generated")

    buf = io.BytesIO()
    sf.write(buf, audio_data, sample_rate, format="WAV")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts_output.wav"},
    )


@app.post("/v1/audio/speech/stream")
async def speech_stream(req: TTSRequest):
    """REST streaming endpoint — Server-Sent Events with PCM chunks.

    Each SSE data line is a JSON object:
      {"type":"audio_chunk","data":"<base64 pcm>","sample_rate":24000,"chunk_index":0}
      {"type":"audio_done","total_duration":3.5,"sample_rate":24000}
      {"type":"error","message":"..."}
    """
    eng = get_engine()

    if req.model_type != eng.model_type:
        await eng.switch_model(req.model_type)

    async def event_stream():
        import base64

        chunk_index = 0
        async for chunk in eng.generate_stream(
            text=req.text,
            language=req.language,
            speaker=req.speaker,
            instruct=req.instruct,
            ref_audio=req.ref_audio,
            ref_text=req.ref_text,
            x_vector_only_mode=req.x_vector_only_mode,
            max_new_tokens=req.max_new_tokens,
        ):
            if chunk["type"] == "audio_chunk":
                pcm_bytes = _numpy_to_pcm16_bytes(chunk["audio"])
                b64 = base64.b64encode(pcm_bytes).decode("ascii")
                yield f"data: {json.dumps({'type': 'audio_chunk', 'data': b64, 'sample_rate': chunk['sample_rate'], 'chunk_index': chunk_index})}\n\n"
                chunk_index += 1
            elif chunk["type"] == "audio_done":
                yield f"data: {json.dumps({'type': 'audio_done', 'total_duration': chunk['total_duration'], 'sample_rate': chunk['sample_rate']})}\n\n"
            elif chunk["type"] == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': chunk['message']})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/audio/speech/cancel")
async def cancel_speech(req: CancelRequest):
    eng = get_engine()
    cancelled = eng.request_cancel(req.request_id)
    if cancelled:
        return {"status": "cancelled", "request_id": req.request_id}
    return {"status": "not_found", "request_id": req.request_id}


@app.websocket("/ws/tts")
async def ws_tts(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS with cancel support.

    Client sends JSON messages:
      {"type":"generate","text":"...","language":"Chinese","speaker":"Vivian",...}
      {"type":"cancel","request_id":"..."}

    Server sends:
      JSON: {"type":"session.start","request_id":"..."}
      Binary: raw PCM int16le audio chunk (24000Hz mono)
      JSON: {"type":"audio.done","total_duration":3.5,"sample_rate":24000}
      JSON: {"type":"error","message":"..."}
    """
    await websocket.accept()
    eng = get_engine()
    request_id = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "cancel":
                rid = msg.get("request_id", "")
                eng.request_cancel(rid)
                await websocket.send_json({"type": "cancel.ack", "request_id": rid})
                continue

            if msg_type == "generate":
                request_id = msg.get("request_id") or os.urandom(4).hex()
                await websocket.send_json({"type": "session.start", "request_id": request_id})
                active_requests[request_id] = websocket

                model_type = msg.get("model_type", eng.model_type)
                if model_type != eng.model_type:
                    await eng.switch_model(model_type)

                try:
                    chunk_idx = 0
                    async for chunk in eng.generate_stream(
                        text=msg.get("text", ""),
                        language=msg.get("language", "Chinese"),
                        speaker=msg.get("speaker"),
                        instruct=msg.get("instruct"),
                        ref_audio=msg.get("ref_audio"),
                        ref_text=msg.get("ref_text"),
                        x_vector_only_mode=msg.get("x_vector_only_mode", False),
                        max_new_tokens=msg.get("max_new_tokens", 2048),
                    ):
                        if chunk["type"] == "audio_chunk":
                            pcm_bytes = _numpy_to_pcm16_bytes(chunk["audio"])
                            await websocket.send_bytes(pcm_bytes)
                            chunk_idx += 1
                        elif chunk["type"] == "audio_done":
                            await websocket.send_json({
                                "type": "audio.done",
                                "total_duration": chunk["total_duration"],
                                "sample_rate": chunk["sample_rate"],
                                "chunks_sent": chunk_idx,
                                "request_id": request_id,
                            })
                        elif chunk["type"] == "error":
                            await websocket.send_json({
                                "type": "error",
                                "message": chunk["message"],
                                "request_id": request_id,
                            })
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected during generation: {request_id}")
                    eng.request_cancel(request_id)
                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    try:
                        await websocket.send_json({"type": "error", "message": str(e), "request_id": request_id})
                    except Exception:
                        pass
                finally:
                    active_requests.pop(request_id, None)
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        if request_id:
            eng.request_cancel(request_id)
            active_requests.pop(request_id, None)
        logger.info("WebSocket client disconnected")


def _numpy_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 numpy array to signed 16-bit LE PCM bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    return pcm16.tobytes()


# Mount static files for the frontend UI
_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=60,
    )
