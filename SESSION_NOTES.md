# Qwen3-TTS-Server 开发会话记录

## 项目概述

基于 vLLM-Omni 的 Qwen3-TTS 流式语音合成服务器。

## 技术决策

### 为什么选 vLLM-Omni 而非原生 vLLM
- 原生 vLLM 不支持 Qwen3-TTS 的流式音频生成
- vLLM-Omni 是 vLLM 的多模态扩展，原生支持 Qwen3-TTS 的两阶段流式架构（Talker AR + Code2Wav）
- 通过 `AsyncOmni` 类提供异步流式接口，每个 stage output 包含部分音频

### 两阶段流式架构
1. **Stage 0 (Talker)**: AR 自回归生成 codec tokens，max_num_batched_tokens=512 实现低首包延迟
2. **Stage 1 (Code2Wav)**: 将 codec tokens 转为 waveform，max_num_batched_tokens=65536 处理长序列

### 通信协议
- **WebSocket** 主通道：二进制帧传 PCM（零开销），JSON 帧传控制消息
- **SSE REST** 备选：base64 编码 PCM
- **非流式 REST** 简单接口：返回完整 WAV

### 模型下载策略
- **下载源优先级**: ModelScope（国内快）→ HuggingFace（备选），支持用户手动选择
- **本地优先**: `models/{model_type}/` 目录存在且完整 → 直接使用
- **下载到项目目录**: 默认 `models/`，支持 `--models-dir` 自定义
- **完整性校验**: 基于 `MODEL_ESSENTIAL_FILES`（10个必要文件）验证

## 项目结构

```
Qwen3-TTS-Server/
├── run.py              # CLI 入口 (argparse + uvicorn)
├── download_models.py  # 独立模型下载脚本
├── requirements.txt    # 运行时依赖
├── .gitignore          # 包含 .venv/, models/
├── server/
│   ├── __init__.py
│   ├── __main__.py     # python -m server 入口
│   ├── config.py       # 常量：音色/语言/模型ID/文件清单/中文描述
│   ├── main.py         # FastAPI 应用（TTS + 模型管理端点）
│   ├── model_manager.py # 模型管理核心（检查/下载/删除/取消）
│   └── vllm_engine.py  # vLLM-Omni 引擎封装
├── static/
│   └── index.html      # 前端单页应用（中文界面，4个Tab）
└── .venv/              # Python 3.12 虚拟环境
```

## API 端点

### TTS 语音合成
| 端点 | 方法 | 说明 |
|------|------|------|
| `/ws/tts` | WebSocket | 流式 TTS + 取消支持 |
| `/v1/audio/speech` | POST | 非流式，返回完整 WAV |
| `/v1/audio/speech/stream` | POST | SSE 流式，base64 PCM |
| `/v1/audio/speech/cancel` | POST | 取消生成 |
| `/v1/audio/speakers` | GET | 音色列表（9个） |
| `/v1/audio/languages` | GET | 语言列表（10种 + Auto） |
| `/v1/audio/models` | GET | 模型 ID 列表 |
| `/v1/audio/status` | GET | 引擎状态 |

### 模型管理
| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/models/status` | GET | 检查所有模型完整性 |
| `/v1/models/{type}/status` | GET | 检查单个模型 |
| `/v1/models/{type}/download` | POST | 开始下载（支持 source 参数） |
| `/v1/models/{type}/cancel-download` | POST | 取消下载 |
| `/v1/models/{type}` | DELETE | 删除模型 |
| `/v1/models/{type}/download-status` | GET | 查询下载进度 |

## 模型信息

三种模型变体共享相同文件结构：

| 模型类型 | HF/ModelScope ID | 中文名 | 说明 |
|----------|-----------------|--------|------|
| custom_voice | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | 预设音色 | 9种精选音色，支持风格指令 |
| voice_design | Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign | 语音设计 | 通过自然语言描述设计音色 |
| base | Qwen/Qwen3-TTS-12Hz-1.7B-Base | 声音克隆 | 3秒参考音频即可克隆 |

### 必要文件（完整性校验用）
`config.json`, `model.safetensors`, `tokenizer_config.json`, `merges.txt`, `vocab.json`, `preprocessor_config.json`, `generation_config.json`, `speech_tokenizer/config.json`, `speech_tokenizer/model.safetensors`, `speech_tokenizer/preprocessor_config.json`

## 环境状态

| 环境 | Python | 包 | 用途 |
|------|--------|-----|------|
| System | 3.12.3 | 无 pip | 系统级 |
| Pyenv | 3.10.12 | basedpyright, pip | LSP 类型检查 |
| Project .venv | 3.12.3 | vllm 0.18.1, vllm-omni 0.18.0, torch 2.10.0, transformers 4.57.6 | 已安装 |

### 版本兼容性
- **vllm-omni 0.18.0 仅兼容 vllm 0.18.x**。vllm 0.19+ 移除了 `vllm.inputs.data`
- `requirements.txt` 已 pin：`vllm>=0.18.0,<0.19.0`

## 已修复

1. **AsyncOmni 初始化错误** — `from_cli_args` 不存在于 vllm-omni 0.18.0 的 `AsyncOmni`/`OmniBase` 类上。改为直接构造 `AsyncOmni(model=model_path, **kwargs)`。`OmniBase.__init__` 接受 `model: str` 和 `**kwargs`（`gpu_memory_utilization`, `stage_configs_path` 等），内部创建 `AsyncOmniEngine`。
2. **FlashInfer 版本不匹配** — `flashinfer-cubin 0.6.8.post1` 与 `flashinfer-python 0.6.6` 不匹配，导致 Stage 0 (Talker) WorkerProc 崩溃。修复：`pip install flashinfer-cubin==0.6.6` 对齐版本，`requirements.txt` 新增 pin。同时添加 `FLASHINFER_DISABLE_VERSION_CHECK=1` 环境变量作为安全回退。
3. **缺失系统依赖** — 安装 `sox`（音频处理）和 `python3.12-dev`（Triton kernel 编译需要 Python.h）。

## 环境注意事项

- **FlashInfer 版本必须匹配**：vllm 0.18.x 需要 `flashinfer-python==0.6.6` + `flashinfer-cubin==0.6.6`，参考 https://github.com/vllm-project/vllm-omni/issues/1946
- **SoX**：Qwen3-TTS 音频加载需要 sox，必须系统安装 `apt install sox`
- **python3.12-dev**：Triton kernel JIT 编译需要 Python.h，否则 SnakeBeta 激活函数回退到 eager 模式
- **WSL 环境**：vLLM 自动检测 WSL 并设置 `pin_memory=False`

## 待完成

1. **实际 GPU 测试** — 代码基于 vLLM-Omni 文档编写，需 GPU 环境验证
2. **模型下载测试** — 验证 ModelScope/HuggingFace 下载是否产生完整可用的模型文件

## 参考资源

- [Qwen3-TTS 官方仓库](https://github.com/QwenLM/Qwen3-TTS/)
- [vLLM-Omni 仓库](https://github.com/vllm-project/vllm-omni)
- [Qwen3Audio 参考项目](https://github.com/RSJWY/Qwen3Audio) — 同步版本参考
