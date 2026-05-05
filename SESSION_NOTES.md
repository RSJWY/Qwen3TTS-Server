# Qwen3-TTS-Server 开发会话记录

## 项目概述

基于 vLLM-Omni 的 Qwen3-TTS 流式语音合成服务器。核心需求：用户已有同步版本的 Qwen3Audio 项目，需要改为流式输出，避免长时间等待。

## 技术决策

### 为什么选 vLLM-Omni 而非原生 vLLM
- 原生 vLLM 不支持 Qwen3-TTS 的流式音频生成
- vLLM-Omni 是 vLLM 的多模态扩展，原生支持 Qwen3-TTS 的两阶段流式架构（Talker AR + Code2Wav）
- 通过 `AsyncOmni` 类提供异步流式接口，每个 stage output 包含部分音频

### 两阶段流式架构
1. **Stage 0 (Talker)**: AR 自回归生成 codec tokens，max_num_batched_tokens=512 实现低首包延迟
2. **Stage 1 (Code2Wav)**: 将 codec tokens 转为 waveform，max_num_batched_tokens=65536 处理长序列

### 通信协议选择
- **WebSocket** 作为主要流式通道：二进制帧传 PCM（零开销），JSON 帧传控制消息
- **SSE REST** 作为备选：base64 编码 PCM，兼容性更好
- **非流式 REST** 作为简单接口：返回完整 WAV

## 项目结构

```
Qwen3-TTS-Server/
├── run.py                  # CLI 入口 (argparse + uvicorn)
├── requirements.txt        # 运行时依赖
├── README.md               # 完整文档
├── .gitignore              # 包含 .venv/
├── server/
│   ├── __init__.py
│   ├── __main__.py         # python -m server 入口
│   ├── config.py           # 音色/语言/模型常量
│   ├── main.py             # FastAPI 应用（所有端点）
│   └── vllm_engine.py      # vLLM-Omni 引擎封装
├── static/
│   └── index.html          # 前端单页应用
└── .venv/                  # Python 3.12 虚拟环境（已创建，待安装依赖）
```

## API 端点

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

## WebSocket 协议

### 客户端 → 服务端
- `{"type":"generate","text":"...","language":"Chinese","speaker":"Vivian",...}` — 开始生成
- `{"type":"cancel","request_id":"..."}` — 取消生成

### 服务端 → 客户端
- JSON `{"type":"session.start","request_id":"..."}` — 会话开始
- 二进制帧 — PCM int16le 24000Hz mono 音频块
- JSON `{"type":"audio.done","total_duration":3.5,"sample_rate":24000}` — 生成完成
- JSON `{"type":"error","message":"..."}` — 错误

## 前端特性

- 3 个 Tab：CustomVoice / VoiceDesign / VoiceClone
- Web Audio API 实时播放 PCM 流
- 取消按钮 + Escape 键取消
- 进度条（基于已接收采样数估算）
- 生成完成后构建 WAV 供下载/回放
- Void Space 调色板（GitHub 风格暗色主题）

## 引擎核心逻辑

### 取消机制
- 每个 `generate_stream` 调用创建 `asyncio.Event` 存入 `_cancel_events[request_id]`
- 每次从 `AsyncOmni.generate()` 迭代时检查 `cancel_event.is_set()`
- 客户端发 cancel 消息 → 服务端调 `engine.request_cancel(request_id)` → set event

### 模型切换
- `switch_model()` 先 unload 当前 AsyncOmni，再加载新模型
- 三种模型类型对应三个 HuggingFace repo：
  - `custom_voice`: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
  - `voice_design`: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
  - `base`: Qwen/Qwen3-TTS-12Hz-1.7B-Base

### prompt_token_ids 估计
- Talker 阶段替换所有输入 embedding，但要求 placeholder 长度匹配
- 调用 `Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information()`
- fallback: 基于文本长度粗估 `(len(text) + len(instruct)) * 1.5 + 256`

## Git 历史

```
c7c56e1 fix: resolve basedpyright reportArgumentType in _extract_sample_rate
520a4b9 docs: add README, fix type annotations, remove unused imports
12d6f0e feat: init Qwen3-TTS streaming server with vLLM-Omni backend
```

## 环境状态

| 环境 | Python | 包 | 用途 |
|------|--------|-----|------|
| System | 3.12.3 | 无 pip/无项目包 | 系统级 |
| Pyenv | 3.10.12 | basedpyright, pip, setuptools | LSP 类型检查 |
| Project .venv | 3.12.3 | 仅 pip | **待安装项目依赖** |

## 待完成

1. **安装项目依赖到 .venv**:
   ```bash
   source .venv/bin/activate
   pip install vllm --torch-backend=auto
   pip install vllm-omni
   pip install -r requirements.txt
   ```

2. **实际 GPU 测试** — 当前代码基于 vLLM-Omni 文档和源码编写，需在有 GPU 的环境验证

3. **基于pyright LSP 诊断** — 剩余 13 个 `reportMissingImports` 错误在安装依赖后会消失

## 参考资源

- [Qwen3-TTS 官方仓库](https://github.com/QwenLM/Qwen3-TTS/)
- [vLLM-Omni 仓库](https://github.com/vllm-project/vllm-omni)
- [Qwen3Audio 参考项目](https://github.com/RSJWY/Qwen3Audio) — 同步版本参考
- vLLM-Omni 部署配置: `vllm_omni/deploy/qwen3_tts.yaml`
- vLLM-Omni 流式客户端示例: `examples/online_serving/text_to_speech/qwen3_tts/streaming_speech_client.py`
