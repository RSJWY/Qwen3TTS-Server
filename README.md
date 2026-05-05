# Qwen3-TTS Server

基于 vLLM-Omni 的 Qwen3-TTS 流式语音合成服务器，支持边生成边播放，首包延迟 ~97ms。

## 架构

```
┌──────────────┐    WebSocket (PCM chunks)    ┌──────────────────┐
│   Frontend    │ ◄──────────────────────────► │   FastAPI Server  │
│  (index.html) │    JSON control messages     │                  │
└──────────────┘                               │  vLLM-Omni       │
                                               │  AsyncOmni       │
                                               └──────────────────┘
```

- **后端**: FastAPI + vLLM-Omni AsyncOmni，真正的流式音频生成
- **前端**: 单页应用，Web Audio API 实时播放 PCM，支持取消
- **协议**: WebSocket 传输二进制 PCM + JSON 控制消息；SSE REST 流式作为备选

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐 Python 3.12）
python3.12 -m venv .venv
source .venv/bin/activate

# 安装 vLLM（必须 0.18.x，与 vllm-omni 0.18.0 兼容）
pip install "vllm>=0.18.0,<0.19.0" --torch-backend=auto

# 安装 vLLM-Omni
pip install vllm-omni

# 安装服务端依赖
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 默认使用 CustomVoice 模型
python run.py

# 指定模型类型
python run.py --model-type voice_design

# 完整参数
python run.py --host 0.0.0.0 --port 8000 --model-type custom_voice --gpu-util 0.3 --device cuda:0

# 或使用模块方式启动
python -m server
```

### 3. 打开浏览器

访问 `http://localhost:8000`

## API 接口

### WebSocket（推荐，流式+取消）

**端点**: `ws://localhost:8000/ws/tts`

**生成请求**:
```json
{
  "type": "generate",
  "text": "你好世界",
  "language": "Chinese",
  "speaker": "Vivian",
  "instruct": "用温柔的语气说",
  "model_type": "custom_voice"
}
```

**取消请求**:
```json
{"type": "cancel", "request_id": "abc12345"}
```

**服务端响应**:
- JSON `{"type":"session.start","request_id":"..."}` — 开始
- 二进制帧 — PCM int16le 音频块 (24000Hz mono)
- JSON `{"type":"audio.done","total_duration":3.5}` — 完成
- JSON `{"type":"error","message":"..."}` — 错误

### REST 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/audio/speech` | POST | 非流式，返回完整 WAV |
| `/v1/audio/speech/stream` | POST | SSE 流式，base64 PCM |
| `/v1/audio/speech/cancel` | POST | 取消生成 |
| `/v1/audio/speakers` | GET | 音色列表 |
| `/v1/audio/languages` | GET | 语言列表 |
| `/v1/audio/models` | GET | 模型 ID 列表 |
| `/v1/audio/status` | GET | 引擎状态 |

**POST 请求体** (speech / speech/stream):
```json
{
  "text": "Hello world",
  "language": "English",
  "speaker": "Ryan",
  "instruct": null,
  "ref_audio": null,
  "ref_text": null,
  "x_vector_only_mode": false,
  "max_new_tokens": 2048,
  "model_type": "custom_voice"
}
```

## 支持的功能

### CustomVoice — 预设音色

9 种精选音色，支持风格指令控制：

| 音色 | 语言 | 描述 |
|------|------|------|
| Vivian | 中文 | 明亮、略带锋芒的年轻女声 |
| Serena | 中文 | 温暖、柔和的年轻女声 |
| Uncle_Fu | 中文 | 低沉醇厚的成熟男声 |
| Dylan | 京味方言 | 清亮自然的京味年轻男声 |
| Eric | 川味方言 | 略带沙哑亮度的成都男声 |
| Ryan | English | 节奏感强的动感男声 |
| Aiden | English | 阳光清澈的美式男中音 |
| Ono_Anna | 日本語 | 轻快俏皮的日系女声 |
| Sohee | 한국어 | 温暖富有情感的韩语女声 |

### VoiceDesign — 语音设计

通过自然语言描述设计任意音色：

```json
{
  "type": "generate",
  "text": "哥哥，你回来啦！",
  "language": "Chinese",
  "instruct": "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显",
  "model_type": "voice_design"
}
```

### VoiceClone — 声音克隆

3 秒参考音频即可克隆声音：

```json
{
  "type": "generate",
  "text": "Hello, nice to meet you.",
  "language": "English",
  "ref_audio": "data:audio/wav;base64,UklGRi...",
  "ref_text": "Transcript of reference audio",
  "model_type": "base"
}
```

## 支持 10 种语言

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

另支持 `Auto` 自动检测语言。

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.12（推荐） |
| GPU | CUDA 计算能力 7.0+ (V100, A100, RTX20xx, L4, H100) |
| GPU 显存 | 8GB+ (1.7B 模型约需 4GB) |
| CUDA | 13.0 兼容（vLLM 0.20.0 默认） |
| 操作系统 | Linux（推荐）/ Windows (WSL2) |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QWEN3_TTS_MODEL_TYPE` | `custom_voice` | 初始模型类型 |
| `QWEN3_TTS_GPU_UTIL` | `0.3` | GPU 显存利用率 |
| `QWEN3_TTS_DEVICE` | `cuda:0` | CUDA 设备 |
| `QWEN3_TTS_STAGE_CONFIGS` | 无 | vLLM-Omni stage 配置 YAML 路径 |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | vLLM 多进程方式（必须） |

## 项目结构

```
Qwen3-TTS-Server/
├── run.py                  # CLI 入口
├── requirements.txt        # Python 依赖
├── server/
│   ├── __init__.py
│   ├── __main__.py         # python -m server 入口
│   ├── config.py           # 音色/语言/模型配置
│   ├── main.py             # FastAPI 应用（所有端点）
│   └── vllm_engine.py      # vLLM-Omni 引擎封装
└── static/
    └── index.html          # 前端单页应用
```

## 故障排除

### vLLM 安装失败

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install vllm --torch-backend=auto
```

### GPU 显存不足

降低 `--gpu-util` 参数：

```bash
python run.py --gpu-util 0.2
```

### Windows 环境

使用 WSL2：

```bash
wsl
cd /path/to/Qwen3-TTS-Server
source .venv/bin/activate
python run.py
```

## 致谢

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/) — 通义千问语音合成模型
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) — vLLM 多模态扩展，原生支持 Qwen3-TTS 流式
- [Qwen3Audio](https://github.com/RSJWY/Qwen3Audio) — 参考项目
