import argparse
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Streaming Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--model-type", default="custom_voice", choices=["custom_voice", "voice_design", "base"],
                        help="Initial model type to load (default: custom_voice)")
    parser.add_argument("--gpu-util", type=float, default=0.3, help="GPU memory utilization 0-1 (default: 0.3)")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument("--stage-configs", default=None, help="Path to vLLM-Omni stage configs YAML")
    args = parser.parse_args()

    os.environ["QWEN3_TTS_MODEL_TYPE"] = args.model_type
    os.environ["QWEN3_TTS_GPU_UTIL"] = str(args.gpu_util)
    os.environ["QWEN3_TTS_DEVICE"] = args.device
    if args.stage_configs:
        os.environ["QWEN3_TTS_STAGE_CONFIGS"] = args.stage_configs

    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=60,
    )


if __name__ == "__main__":
    main()
