"""
GPT-SoVITS API 实现

### 目录结构
GPT-SoVITS/
├── api_114514.py                # 本文件, API 主程序
├── GPT_SoVITS/                  # GPT-SoVITS 核心库
│   └── configs/
│       └── tts_infer.yaml       # 默认配置文件
├── roles/                       # 角色配置目录
│   ├── role1/                   # 示例角色 role1
│   │   ├── tts_infer.yaml       # 角色配置文件（可选）
│   │   ├── model.ckpt           # GPT 模型（可选）
│   │   ├── model.pth            # SoVITS 模型（可选）
│   │   └── audio/               # 角色音频目录
│   │       ├── zh/              # 中文音频
│   │       │   ├── [开心]voice1.wav  # 参考音频文件，每个角色必有至少一个
│   │       │   ├── [开心]voice1.txt  # 文本文件，可选，用于对[开心]voice1.wav提供音频参考
│   │       ├── jp/              # 日文音频
│   │       │   ├── [开心]voice2.wav
│   │       │   ├── [开心]voice2.txt
│   ├── role2/
│   │   ├── tts_infer.yaml
│   │   ├── model.ckpt
│   │   ├── model.pth
│   │   └── audio/
│   │       ├── zh/
│   │       │   ├── [开心]voice1.wav
│   │       │   ├── [开心]voice1.txt
│   │       │   ├── [悲伤]asdafasdas.wav
│   │       │   ├── [悲伤]asdafasdas.txt
│   │       ├── jp/
│   │       │   ├── [开心]voice2.wav
│   │       │   ├── [开心]voice2.txt

### 完整请求示例 (/ttsrole POST)
以下是一个包含所有参数的 POST 请求示例，发送到 http://127.0.0.1:9880/ttsrole
{
    "text": "你好",                     # str, 必填, 要合成的文本内容
    "role": "role1",                   # str, 必填, 角色名称，决定使用 roles/{role} 中的配置和音频
    "emotion": "开心",                  # str, 可选, 情感标签，用于从 roles/{role}/audio 中选择音频
    "text_lang": "jp",                 # str, 可选, 默认 "zh", 文本语言，必须在 languages 中支持, 去tts_infer.yaml里看
    "ref_audio_path": "/path/to/ref.wav",  # str, 可选, 参考音频路径，若提供则优先使用，跳过自动选择
    "aux_ref_audio_paths": ["/path1.wav", "/path2.wav"],  # List[str], 可选, 辅助参考音频路径，用于多说话人融合
    "prompt_lang": "jp",               # str, 可选, 提示文本语言，若提供 ref_audio_path 则需指定
    "prompt_text": "こんにちは",       # str, 可选, 提示文本，与 ref_audio_path 配对使用
    "top_k": 10,                       # int, 可选, Top-K 采样值，覆盖 inference.top_k
    "top_p": 0.8,                      # float, 可选, Top-P 采样值，覆盖 inference.top_p
    "temperature": 1.0,                # float, 可选, 温度值，覆盖 inference.temperature
    "text_split_method": "cut5",       # str, 可选, 文本分割方法，覆盖 inference.text_split_method, 具体见text_segmentation_method.py
    "batch_size": 2,                   # int, 可选, 批处理大小，覆盖 inference.batch_size
    "batch_threshold": 0.75,           # float, 可选, 批处理阈值，覆盖 inference.batch_threshold
    "split_bucket": true,              # bool, 可选, 是否按桶分割，覆盖 inference.split_bucket
    "speed_factor": 1.2,               # float, 可选, 语速因子，覆盖 inference.speed_factor
    "fragment_interval": 0.3,          # float, 可选, 片段间隔（秒），覆盖 inference.fragment_interval
    "seed": 42,                        # int, 可选, 随机种子，覆盖 seed
    "media_type": "wav",               # str, 可选, 默认 "wav", 输出格式，支持 "wav", "raw", "ogg", "aac"
    "streaming_mode": false,           # bool, 可选, 默认 false, 是否流式返回
    "parallel_infer": true,            # bool, 可选, 默认 true, 是否并行推理
    "repetition_penalty": 1.35,        # float, 可选, 重复惩罚值，覆盖 inference.repetition_penalty
    "version": "v2",                   # str, 可选, 配置文件版本，覆盖 version
    "languages": ["zh", "jp", "en"],   # List[str], 可选, 支持的语言列表，覆盖 languages
    "t2s_model_path": "/path/to/gpt.ckpt",  # str, 可选, GPT 模型路径，覆盖 t2s_model.path
    "t2s_model_type": "bert",          # str, 可选, GPT 模型类型，覆盖 t2s_model.model_type
    "t2s_model_device": "cpu",         # str, 可选, GPT 模型设备，覆盖 t2s_model.device，默认检测显卡
    "vits_model_path": "/path/to/sovits.pth",  # str, 可选, SoVITS 模型路径，覆盖 vits_model.path
    "vits_model_device": "cpu"         # str, 可选, SoVITS 模型设备，覆盖 vits_model.device，默认检测显卡
}

### 参数必要性和优先级
- 必填参数: text, role（仅 /ttsrole）
- 可选参数: 其他均为可选，默认值从 roles/{role}/tts_infer.yaml 或 GPT_SoVITS/configs/tts_infer.yaml 获取
- 优先级: POST 请求参数 > roles/{role}/tts_infer.yaml > 默认 GPT_SoVITS/configs/tts_infer.yaml
  - 例如: 若提供 "t2s_model_device": "cpu"，即使检测到显卡，也使用 CPU
  - 若未提供 "ref_audio_path"，则根据 role、text_lang、emotion 从 roles/{role}/audio 自动选择

### 讲解
1. 必填参数:
   - text: 合成文本，核心输入
   - role: 指定角色，决定配置和音频来源，/ttsrole 独有
2. 音频选择:
   - 若提供 ref_audio_path，则使用它
   - 否则根据 role、text_lang、emotion 从 roles/{role}/audio/{text_lang} 中选择
   - emotion 匹配 [emotion] 前缀音频，未匹配则随机选择
3. 设备选择:
   - 默认尝试检测显卡（torch.cuda.is_available()），若可用则用 "cuda"，否则 "cpu"
   - 若缺少 torch 依赖或检测失败，回退到 "cpu"
   - POST 参数 t2s_model_device 和 vits_model_device 可强制指定设备，优先级最高
4. 配置文件:
   - 默认加载 GPT_SoVITS/configs/tts_infer.yaml
   - 若 roles/{role}/tts_infer.yaml 存在且未被请求参数覆盖，则使用它
   - 请求参数（如 top_k）覆盖所有配置文件
5. 返回格式:
   - 成功时返回 JSON，包含 Base64 编码的音频数据
   - 失败时返回 JSON，包含错误消息和可能的异常详情
6. 运行:
   - python api_114514.py -a 127.0.0.1 -p 9880
   - 检查启动日志确认设备
   
### 响应示例
1. 成功生成音频
{
    "status": "success",
    "message": "Audio generated successfully",
    "media_type": "wav",
    "audio_data": "UklGRi...（Base64 编码的音频数据）"
}
- 状态码: 200
- audio_data: Base64 编码的二进制音频数据
2. 缺少必填参数 text
{
    "status": "error",
    "message": "text is required"
}
- 状态码: 400
3. 缺少必填参数 role
{
    "status": "error",
    "message": "role is required for /ttsrole"
}
- 状态码: 400
4. 角色目录不存在且无参考音频
{
    "status": "error",
    "message": "Role directory not found and no suitable reference audio provided"
}
- 状态码: 400
5. 运行时异常
{
    "status": "error",
    "message": "tts failed",
    "exception": "CUDA out of memory"
}
- 状态码: 400

"""

import os
import sys
import traceback
from typing import Generator, Optional, List, Dict
import random
import glob
import base64  # 新增 Base64 编码支持

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names

# 尝试导入 PyTorch，检测显卡支持
try:
    import torch
    cuda_available = torch.cuda.is_available()
except ImportError:
    cuda_available = False
    print("缺少 PyTorch 依赖，默认使用 CPU")
except Exception as e:
    cuda_available = False
    print(f"检测显卡时出错: {str(e)}，默认使用 CPU")

i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT_SoVITS/configs/tts_infer.yaml"

default_device = "cuda" if cuda_available else "cpu"
print(f"默认设备设置为: {default_device}")

# 初始化 TTS 配置
tts_config = TTS_Config(config_path)
if "device" not in tts_config.t2s_model:
    tts_config.t2s_model["device"] = default_device
if "device" not in tts_config.vits_model:
    tts_config.vits_model["device"] = default_device
tts_pipeline = TTS(tts_config)

APP = FastAPI()

class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: List[str] = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = None
    top_p: float = None
    temperature: float = None
    text_split_method: str = None
    batch_size: int = None
    batch_threshold: float = None
    split_bucket: bool = None
    speed_factor: float = None
    fragment_interval: float = None
    seed: int = None
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = None
    version: str = None
    languages: List[str] = None
    t2s_model_path: str = None
    t2s_model_type: str = None
    t2s_model_device: str = None
    vits_model_path: str = None
    vits_model_device: str = None

class TTS1_Request(BaseModel):
    text: str = None
    text_lang: str = "zh"
    ref_audio_path: str = None
    aux_ref_audio_paths: List[str] = None
    prompt_lang: str = None
    prompt_text: str = None
    emotion: str = None
    role: str = None
    top_k: int = None
    top_p: float = None
    temperature: float = None
    text_split_method: str = None
    batch_size: int = None
    batch_threshold: float = None
    split_bucket: bool = None
    speed_factor: float = None
    fragment_interval: float = None
    seed: int = None
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = None
    version: str = None
    languages: List[str] = None
    t2s_model_path: str = None
    t2s_model_type: str = None
    t2s_model_device: str = None
    vits_model_path: str = None
    vits_model_device: str = None

def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer

def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer

def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer

def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen([
        'ffmpeg', '-f', 's16le', '-ar', str(rate), '-ac', '1', '-i', 'pipe:0',
        '-c:a', 'aac', '-b:a', '192k', '-vn', '-f', 'adts', 'pipe:1'
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer

def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer

def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()

def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)

def check_params(req: dict, is_tts1: bool = False):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "zh" if is_tts1 else "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")
    
    if is_tts1:
        role: str = req.get("role", "")
        if role in [None, ""]:
            return {"status": "error", "message": "role is required for /ttsrole"}
    else:
        if ref_audio_path in [None, ""]:
            return {"status": "error", "message": "ref_audio_path is required"}
        if prompt_lang in [None, ""]:
            return {"status": "error", "message": "prompt_lang is required"}
    
    if text in [None, ""]:
        return {"status": "error", "message": "text is required"}
    
    languages = req.get("languages") or tts_config.languages
    if text_lang.lower() not in languages:
        return {"status": "error", "message": f"text_lang: {text_lang} is not supported"}
    if not is_tts1 and prompt_lang.lower() not in languages:
        return {"status": "error", "message": f"prompt_lang: {prompt_lang} is not supported"}
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return {"status": "error", "message": f"media_type: {media_type} is not supported"}
    elif media_type == "ogg" and not streaming_mode:
        return {"status": "error", "message": "ogg format is not supported in non-streaming mode"}
    if text_split_method not in cut_method_names:
        return {"status": "error", "message": f"text_split_method: {text_split_method} is not supported"}
    
    return None

def load_role_config(role: str, req: dict):
    role_dir = os.path.join(now_dir, "roles", role)
    if not os.path.exists(role_dir):
        return False
    
    if not req.get("version"):
        config_path_new = os.path.join(role_dir, "tts_infer.yaml")
        if os.path.exists(config_path_new):
            global tts_config, tts_pipeline
            tts_config = TTS_Config(config_path_new)
            tts_pipeline = TTS(tts_config)
    
    if not req.get("t2s_model_path"):
        gpt_path = glob.glob(os.path.join(role_dir, "*.ckpt"))
        if gpt_path:
            tts_pipeline.init_t2s_weights(gpt_path[0])
    
    if not req.get("vits_model_path"):
        sovits_path = glob.glob(os.path.join(role_dir, "*.pth"))
        if sovits_path:
            tts_pipeline.init_vits_weights(sovits_path[0])
    
    return True

def select_ref_audio(role: str, text_lang: str, emotion: str = None):
    audio_base_dir = os.path.join(now_dir, "roles", role, "audio")
    if not os.path.exists(audio_base_dir):
        return None, None, None
    
    lang_dir = os.path.join(audio_base_dir, text_lang.lower())
    all_langs = [d for d in os.listdir(audio_base_dir) if os.path.isdir(os.path.join(audio_base_dir, d))]
    
    def find_audio_in_dir(dir_path):
        if not os.path.exists(dir_path):
            return None, None
        audio_files = glob.glob(os.path.join(dir_path, "[*]*.*"))
        if not audio_files:
            audio_files = glob.glob(os.path.join(dir_path, "*.*"))
        if not audio_files:
            return None, None
        
        if emotion:
            emotion_files = [f for f in audio_files if f"[{emotion}]" in os.path.basename(f)]
            if emotion_files:
                audio_path = random.choice(emotion_files)
            else:
                audio_path = random.choice(audio_files)
        else:
            audio_path = random.choice(audio_files)
        
        txt_path = audio_path.rsplit(".", 1)[0] + ".txt"
        prompt_text = None
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
        
        return audio_path, prompt_text
    
    audio_path, prompt_text = find_audio_in_dir(lang_dir)
    if audio_path:
        return audio_path, prompt_text, text_lang.lower()
    
    for lang in all_langs:
        if lang != text_lang.lower():
            audio_path, prompt_text = find_audio_in_dir(os.path.join(audio_base_dir, lang))
            if audio_path:
                return audio_path, prompt_text, lang
    
    return None, None, None

async def tts_handle(req: dict, is_tts1: bool = False):
    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")
    
    check_res = check_params(req, is_tts1)
    if check_res is not None:
        return JSONResponse(status_code=400, content=check_res)
    
    role_exists = False
    if is_tts1:
        role_exists = load_role_config(req["role"], req)
        
        if not req.get("ref_audio_path"):
            ref_audio_path, prompt_text, prompt_lang = select_ref_audio(req["role"], req["text_lang"], req.get("emotion"))
            if ref_audio_path:
                req["ref_audio_path"] = ref_audio_path
                req["prompt_text"] = prompt_text or ""
                req["prompt_lang"] = prompt_lang
            elif not role_exists:
                return JSONResponse(status_code=400, content={"status": "error", "message": "Role directory not found and no suitable reference audio provided"})
    
    # 应用请求中的 YAML 参数，优先级最高
    if req.get("version"):
        tts_config.version = req["version"]
    if req.get("languages"):
        tts_config.languages = req["languages"]
    if req.get("t2s_model_path"):
        tts_config.t2s_model["path"] = req["t2s_model_path"]
        tts_pipeline.init_t2s_weights(req["t2s_model_path"])
    if req.get("t2s_model_type"):
        tts_config.t2s_model["model_type"] = req["t2s_model_type"]
    if req.get("t2s_model_device"):
        tts_config.t2s_model["device"] = req["t2s_model_device"]
    if req.get("vits_model_path"):
        tts_config.vits_model["path"] = req["vits_model_path"]
        tts_pipeline.init_vits_weights(req["vits_model_path"])
    if req.get("vits_model_device"):
        tts_config.vits_model["device"] = req["vits_model_device"]
    
    inference = tts_config.inference
    if req.get("top_k") is not None:
        inference["top_k"] = req["top_k"]
    if req.get("top_p") is not None:
        inference["top_p"] = req["top_p"]
    if req.get("temperature") is not None:
        inference["temperature"] = req["temperature"]
    if req.get("text_split_method"):
        inference["text_split_method"] = req["text_split_method"]
    if req.get("batch_size") is not None:
        inference["batch_size"] = req["batch_size"]
    if req.get("batch_threshold") is not None:
        inference["batch_threshold"] = req["batch_threshold"]
    if req.get("split_bucket") is not None:
        inference["split_bucket"] = req["split_bucket"]
    if req.get("speed_factor") is not None:
        inference["speed_factor"] = req["speed_factor"]
    if req.get("fragment_interval") is not None:
        inference["fragment_interval"] = req["fragment_interval"]
    if req.get("repetition_penalty") is not None:
        inference["repetition_penalty"] = req["repetition_penalty"]
    if req.get("seed") is not None:
        tts_config.seed = req["seed"]
    
    if streaming_mode:
        req["return_fragment"] = True
    
    try:
        tts_generator = tts_pipeline.run(req)
        
        if streaming_mode:
            # 流式模式暂不支持 Base64，返回错误
            return JSONResponse(status_code=400, content={"status": "error", "message": "Streaming mode is not supported with Base64 response"})
        else:
            sr, audio_data = next(tts_generator)
            audio_bytes = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            response_json = {
                "status": "success",
                "message": "Audio generated successfully",
                "media_type": media_type,
                "audio_data": audio_base64
            }
            return JSONResponse(status_code=200, content=response_json)
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": "tts failed", "exception": str(e)})

@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "command is required"})
    handle_control(command)

@APP.get("/tts")
async def tts_get_endpoint(
    text: str = None, text_lang: str = None, ref_audio_path: str = None,
    aux_ref_audio_paths: List[str] = None, prompt_lang: str = None, prompt_text: str = "",
    top_k: int = 5, top_p: float = 1, temperature: float = 1, text_split_method: str = "cut0",
    batch_size: int = 1, batch_threshold: float = 0.75, split_bucket: bool = True,
    speed_factor: float = 1.0, fragment_interval: float = 0.3, seed: int = -1,
    media_type: str = "wav", streaming_mode: bool = False, parallel_infer: bool = True,
    repetition_penalty: float = 1.35
):
    req = {
        "text": text,
        "text_lang": text_lang.lower() if text_lang else None,
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_lang": prompt_lang.lower() if prompt_lang else None,
        "prompt_text": prompt_text,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": batch_size,
        "batch_threshold": batch_threshold,
        "split_bucket": split_bucket,
        "speed_factor": speed_factor,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty
    }
    return await tts_handle(req)

@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    if req["text_lang"]:
        req["text_lang"] = req["text_lang"].lower()
    if req["prompt_lang"]:
        req["prompt_lang"] = req["prompt_lang"].lower()
    return await tts_handle(req)

@APP.get("/ttsrole")
async def tts1_get_endpoint(
    text: str = None, text_lang: str = "zh", ref_audio_path: str = None,
    aux_ref_audio_paths: List[str] = None, prompt_lang: str = None, prompt_text: str = None,
    emotion: str = None, role: str = None, top_k: int = 5, top_p: float = 1,
    temperature: float = 1, text_split_method: str = "cut5", batch_size: int = 1,
    batch_threshold: float = 0.75, split_bucket: bool = True, speed_factor: float = 1.0,
    fragment_interval: float = 0.3, seed: int = -1, media_type: str = "wav",
    streaming_mode: bool = False, parallel_infer: bool = True, repetition_penalty: float = 1.35
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_lang": prompt_lang.lower() if prompt_lang else None,
        "prompt_text": prompt_text,
        "emotion": emotion,
        "role": role,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": batch_size,
        "batch_threshold": batch_threshold,
        "split_bucket": split_bucket,
        "speed_factor": speed_factor,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty
    }
    return await tts_handle(req, is_tts1=True)

@APP.post("/ttsrole")
async def tts1_post_endpoint(request: TTS1_Request):
    req = request.dict()
    if req["text_lang"]:
        req["text_lang"] = req["text_lang"].lower()
    if req["prompt_lang"]:
        req["prompt_lang"] = req["prompt_lang"].lower()
    return await tts_handle(req, is_tts1=True)

@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"status": "error", "message": "gpt weight path is required"})
        tts_pipeline.init_t2s_weights(weights_path)
        return JSONResponse(status_code=200, content={"status": "success", "message": "success"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"change gpt weight failed", "exception": str(e)})

@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"status": "error", "message": "sovits weight path is required"})
        tts_pipeline.init_vits_weights(weights_path)
        return JSONResponse(status_code=200, content={"status": "success", "message": "success"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"change sovits weight failed", "exception": str(e)})

@APP.get("/set_refer_audio")
async def set_refer_audio(refer_audio_path: str = None):
    try:
        tts_pipeline.set_ref_audio(refer_audio_path)
        return JSONResponse(status_code=200, content={"status": "success", "message": "success"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"set refer audio failed", "exception": str(e)})

if __name__ == "__main__":
    try:
        if host == 'None':   # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
