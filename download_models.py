"""Standalone model download script for Qwen3-TTS-Server.

Usage:
    python download_models.py                    # Download all models
    python download_models.py custom_voice       # Download specific model(s)
    python download_models.py --list             # List model status
    python download_models.py --delete base      # Delete a model
    python download_models.py --dir /path/models # Custom models directory
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Add project root to path so we can import server.config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.config import MODEL_IDS, MODEL_DESCRIPTIONS, MODEL_ESSENTIAL_FILES
from server.model_manager import ModelManager


def list_models(models_dir: str) -> None:
    print(f"\n模型目录: {os.path.abspath(models_dir)}\n")
    print(f"{'模型类型':<15} {'状态':<16} {'大小':<10} {'描述'}")
    print("-" * 70)
    for mt in MODEL_IDS:
        local_path = os.path.join(models_dir, mt)
        exists = os.path.isdir(local_path)
        if exists:
            missing = [
                f
                for f in MODEL_ESSENTIAL_FILES
                if not os.path.isfile(os.path.join(local_path, f))
            ]
            total_size = sum(
                os.path.getsize(os.path.join(local_path, f))
                for f in MODEL_ESSENTIAL_FILES
                if os.path.isfile(os.path.join(local_path, f))
            )
            size_mb = round(total_size / 1048576, 1)
            if missing:
                status = f"不完整({len(missing)}缺失)"
            else:
                status = "✓ 完整"
        else:
            size_mb = 0
            status = "✗ 未下载"
        desc = MODEL_DESCRIPTIONS.get(mt, "")
        print(f"{mt:<15} {status:<16} {size_mb:>6} MB  {desc}")
    print()


def delete_model(models_dir: str, model_type: str) -> None:
    local_path = os.path.join(models_dir, model_type)
    if os.path.isdir(local_path):
        import shutil

        shutil.rmtree(local_path)
        print(f"已删除: {local_path}")
    else:
        print(f"模型不存在: {local_path}")


def download_models(models_dir: str, model_types: list[str], source: str = "modelscope") -> None:
    mgr = ModelManager(models_dir=models_dir)

    for mt in model_types:
        if mt not in MODEL_IDS:
            print(f"未知模型类型: {mt}")
            print(f"可选: {', '.join(MODEL_IDS.keys())}")
            sys.exit(1)

    for mt in model_types:
        info = mgr.check_model(mt)
        if info["complete"]:
            print(f"[{mt}] 已存在且完整，跳过 ({info['total_size_mb']} MB)")
            continue

        print(f"\n[{mt}] 开始下载...")
        if info["exists"] and info["missing_files"]:
            print(f"  缺失文件: {', '.join(info['missing_files'])}")

        result = mgr.download_model(mt, source=source)
        if result["status"] == "already_complete":
            print(f"[{mt}] 已完整")
            continue

        while True:
            status = mgr.get_download_status(mt)
            msg = status["download_message"]
            progress = status["download_progress"]
            ds = status["download_status"]
            print(f"\r  [{progress:>3}%] {msg}    ", end="", flush=True)
            if ds in ("complete", "error", "cancelled"):
                print()
                break
            time.sleep(0.5)

        final = mgr.get_download_status(mt)
        if final["model_complete"]:
            size = mgr.check_model(mt)["total_size_mb"]
            print(f"[{mt}] ✓ 下载完成 ({size} MB)")
        else:
            print(f"[{mt}] ✗ 下载失败: {final['download_message']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-TTS 模型管理工具")
    parser.add_argument(
        "model_type",
        nargs="*",
        help="要下载的模型类型（custom_voice, voice_design, base）",
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "huggingface", "auto"],
        default="modelscope",
        help="下载源（默认: modelscope）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_models",
        help="列出模型状态",
    )
    parser.add_argument(
        "--delete",
        metavar="MODEL_TYPE",
        help="删除指定模型",
    )
    parser.add_argument(
        "--dir",
        default="models",
        dest="models_dir",
        help="模型存储目录（默认: models）",
    )
    args = parser.parse_args()

    models_dir = args.models_dir

    if args.list_models:
        list_models(models_dir)
    elif args.delete:
        delete_model(models_dir, args.delete)
    else:
        model_types = args.model_type or list(MODEL_IDS.keys())
        download_models(models_dir, model_types, args.source)


if __name__ == "__main__":
    main()
