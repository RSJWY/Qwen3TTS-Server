"""Configuration constants for Qwen3-TTS Server."""

# Local models directory (relative to project root by default)
DEFAULT_MODELS_DIR = "models"

# HuggingFace model IDs used by vLLM-Omni directly
MODEL_IDS = {
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

# ModelScope IDs mirror HuggingFace IDs for Qwen models
MODELSCOPE_IDS = {
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

MODEL_DESCRIPTIONS = {
    "custom_voice": "预设音色模型 — 9种精选音色，支持风格指令",
    "voice_design": "语音设计模型 — 通过自然语言描述设计音色",
    "base": "声音克隆基础模型 — 3秒参考音频即可克隆",
}

# Essential files that must exist for a model to be considered complete.
# All three model variants share the same file structure.
MODEL_ESSENTIAL_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "preprocessor_config.json",
    "generation_config.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
]

MODEL_OPTIONAL_FILES = [
    "README.md",
    ".gitattributes",
    "speech_tokenizer/configuration.json",
]

DEFAULT_MODEL_TYPE = "custom_voice"
DEFAULT_SAMPLE_RATE = 24000

SPEAKERS = {
    "Vivian": {
        "zh": "明亮、略带锋芒的年轻女声",
        "en": "Bright, slightly edgy young female voice",
        "language": "Chinese",
    },
    "Serena": {
        "zh": "温暖、柔和的年轻女声",
        "en": "Warm, gentle young female voice",
        "language": "Chinese",
    },
    "Uncle_Fu": {
        "zh": "低沉醇厚的成熟男声",
        "en": "Seasoned male voice with a low, mellow timbre",
        "language": "Chinese",
    },
    "Dylan": {
        "zh": "清亮自然的京味年轻男声",
        "en": "Youthful Beijing male voice with a clear, natural timbre",
        "language": "Chinese/Beijing Dialect",
    },
    "Eric": {
        "zh": "略带沙哑亮度的成都男声",
        "en": "Lively Chengdu male voice with a slightly husky brightness",
        "language": "Chinese/Sichuan Dialect",
    },
    "Ryan": {
        "zh": "节奏感强的动感男声",
        "en": "Dynamic male voice with strong rhythmic drive",
        "language": "English",
    },
    "Aiden": {
        "zh": "阳光清澈的美式男中音",
        "en": "Sunny American male voice with a clear midrange",
        "language": "English",
    },
    "Ono_Anna": {
        "zh": "轻快俏皮的日系女声",
        "en": "Playful Japanese female voice with a light, nimble timbre",
        "language": "Japanese",
    },
    "Sohee": {
        "zh": "温暖富有情感的韩语女声",
        "en": "Warm Korean female voice with rich emotion",
        "language": "Korean",
    },
}

LANGUAGES = [
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

# Valid language values (including Auto)
VALID_LANGUAGES = ["Auto"] + LANGUAGES

# Server defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# GPU memory utilization for vLLM stages
DEFAULT_GPU_MEMORY_UTILIZATION = 0.3

# vLLM-Omni stage configs reference:
# Stage 0 (talker): max_num_batched_tokens=512 for low first-chunk latency
# Stage 1 (code2wav): max_num_batched_tokens=65536 for codec prefill length
# codec_chunk_frames: 25, codec_left_context_frames: 72
