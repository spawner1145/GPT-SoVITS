"""
# WebAPI文档

`python api_role_v3.py -a 127.0.0.1 -p 9880`

## 执行参数:
    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`

### 配置文件:
    在 api-config.yaml 配置需要加载的模型路径

### 示例：
    在 api-example 里有调用示例代码
    在 main 当中也游示例

## 调用:

### 推理

endpoint: `/tts`
GET:
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&media_type=wav&streaming_mode=True

可用参数：
Parameters:
- text: str,                   # str.(required) text to be synthesized
- text_lang: str = "auto",    # str.(required) language of the text to be synthesized, "auto" for auto-detection
- ref_audio_path: str,        # str.(required) reference audio path
- prompt_lang: str,           # str.(required) language of the prompt text for the reference audio
- prompt_text: str = "",      # str.(optional) prompt text for the reference audio
- top_k: int = 5,             # int. top k sampling
- top_p: float = 1,           # float. top p sampling
- temperature: float = 1,      # float. temperature for sampling
- sample_steps: int = 16,      # int. When you use v3 model, you can set this sample_steps
- media_type: str = "wav",     # str. Set the file format for returning audio
- streaming_mode: bool = False, # bool. whether to return a streaming response
- threshold: int = 30          # int. Text segmentation parameter, the lower value, the faster the streaming inference, but the worse the audio quality

### 角色推理

endpoint: `/ttsrole`
GET:
http://127.0.0.1:9880/ttsrole?text=你好&role=role1&emotion=开心&text_lang=zh


可用参数：
Parameters:
- text: str,                   # str.(required) text to be synthesized
- role: str,                   # str.(required) role name, determines config and audio from roles/{role}
- text_lang: str = "zh",      # str.(optional) language of the text to be synthesized, default "zh", "auto" for auto-detection
- ref_audio_path: str = None, # str.(optional) reference audio path, overrides auto-selection if provided
- prompt_lang: str = None,     # str.(optional) language of the prompt text, required if ref_audio_path is provided, auto-set if text_lang="auto"
- prompt_text: str = None,     # str.(optional) prompt text for the reference audio
- emotion: str = None,         # str.(optional) emotion tag to select audio from roles/{role}/reference_audios
- top_k: int = 5,             # int. top k sampling
- top_p: float = 1,           # float. top p sampling
- temperature: float = 1,      # float. temperature for sampling
- sample_steps: int = 16,      # int. When you use v3 model, you can set this sample_steps
- media_type: str = "wav",     # str. Set the file format for returning audio
- streaming_mode: bool = False, # bool. whether to return a streaming response
- threshold: int = 30          # int. Text segmentation parameter, the lower value, the faster the streaming inference, but the worse the audio quality

### 切换GPT模型
endpoint: `/set_gpt_weights`
GET:
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt


### 切换SoVITS模型
endpoint: `/set_sovits_weights`
GET:
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth

## 优先级和逻辑:
- 配置文件优先级: POST 请求参数 > roles/{role}/tts_infer.yaml > api-config.yaml
- /ttsrole 音频选择逻辑:
  - 若提供 ref_audio_path，则使用它，prompt_lang 必须指定，prompt_text 从同名 .txt 文件读取，若无则从文件名提取
  - 若未提供 ref_audio_path：
    - text_lang != "auto": 从 roles/{role}/reference_audios/{text_lang} 中选择匹配 emotion 的音频，若无则随机选择，prompt_lang 设为 text_lang
    - text_lang = "auto": 在 roles/{role}/reference_audios 下所有语言文件夹中寻找匹配 emotion 的音频，若找到则 prompt_lang 为该语言文件夹名，若无则随机选择并以其语言文件夹名作为 prompt_lang
  - prompt_text 选择逻辑:
    - 若存在同名 .txt 文件，读取其内容
    - 若无 .txt 文件且文件名含 "【xxx】"，提取 "xxx" 后的部分（如 "【开心】abc.wav" -> "abc"）
    - 否则取文件名去掉后缀的部分（如 "ref.wav" -> "ref"）
"""

import argparse
import os
import signal
import sys
import traceback
import yaml
import uvicorn
import subprocess
import numpy as np
import soundfile as sf

from io import BytesIO
from typing import Generator, Optional, List, Dict
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from gpt_sovits.infer import GPTSoVITSInference

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

default_device = "cuda" if cuda_available else "cpu"
print(f"默认设备设置为: {default_device}")

parser = argparse.ArgumentParser(description="GPT-SoVITS-api")
parser.add_argument("-p", "--port", type=int, default=9880, help="server port")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="server ip")
args = parser.parse_args()

def load_config():
    config_path = "./api-config.yaml"
    default_config = {
        "device": default_device,
        "is_half": False,
        "version": "v2",
        "t2s_weights_path": "pretrained_models/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "vits_weights_path": "pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
        "bert_base_path": "pretrained_models/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": "pretrained_models/chinese-hubert-base",
    }

    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        print(f"Config file created at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("Config loaded successfully:", config)
    return config

def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer

def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer

def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with BytesIO() as wav_buf:
        sf.write(wav_buf, data, rate, format='wav')
        wav_buf.seek(0)
        return wav_buf

def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen([
        'ffmpeg', '-f', 's16le', '-ar', str(rate), '-ac', '1', '-i', 'pipe:0',
        '-c:a', 'aac', '-b:a', '192k', '-vn', '-f', 'adts', 'pipe:1'
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer

def pack_audio(data: np.ndarray, rate: int, media_type: str) -> BytesIO:
    io_buffer = BytesIO()
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

def check_params(text: str,
                 text_lang: str,
                 ref_audio_path: str,
                 prompt_lang: str,
                 prompt_text: str,
                 media_type: str,
                 streaming_mode: bool):
    if not text:
        return JSONResponse(status_code=400, content={"error": "text is required"})
    if not text_lang:
        return JSONResponse(status_code=400, content={"error": "text_lang is required"})
    if text_lang == "auto":
        print("Warning: text_lang is set to 'auto'")
    if not ref_audio_path:
        return JSONResponse(status_code=400, content={"error": "ref_audio_path is required"})
    if not prompt_lang:
        return JSONResponse(status_code=400, content={"error": "prompt_lang is required"})
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"error": f"media_type: {media_type} is not supported"})
    if media_type == "ogg" and not streaming_mode:
        return JSONResponse(status_code=400, content={"error": "ogg format is not supported in non-streaming mode"})
    return None

def check_ttsrole_params(text: str, role: str):
    if not text:
        return JSONResponse(status_code=400, content={"error": "text is required"})
    if not role:
        return JSONResponse(status_code=400, content={"error": "role is required"})
    return None

def load_role_config(role: str, base_config: dict):
    role_dir = os.path.join(now_dir, "roles", role)
    if not os.path.exists(role_dir):
        return False, base_config
    
    config_path_new = os.path.join(role_dir, "tts_infer.yaml")
    if os.path.exists(config_path_new):
        with open(config_path_new, "r", encoding="utf-8") as f:
            role_config = yaml.safe_load(f)
        print(f"Loaded role config from {config_path_new}: {role_config}")
        return True, role_config
    
    return True, base_config

def select_ref_audio(role: str, text_lang: str, emotion: str = None):
    audio_base_dir = os.path.join(now_dir, "roles", role, "reference_audios")
    if not os.path.exists(audio_base_dir):
        return None, None, None
    
    if text_lang != "auto":
        lang_dir = os.path.join(audio_base_dir, text_lang.lower())
        all_langs = [d for d in os.listdir(audio_base_dir) if os.path.isdir(os.path.join(audio_base_dir, d))]
        
        def find_audio_in_dir(dir_path):
            if not os.path.exists(dir_path):
                return None, None
            audio_files = glob.glob(os.path.join(dir_path, "【*】*.*"))
            if not audio_files:
                audio_files = glob.glob(os.path.join(dir_path, "*.*"))
            if not audio_files:
                return None, None
            
            if emotion:
                emotion_files = [f for f in audio_files if f"【{emotion}】" in os.path.basename(f)]
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
            else:
                basename = os.path.basename(audio_path)
                start_idx = basename.find("】") + 1
                end_idx = basename.rfind(".")
                if start_idx > 0 and end_idx > start_idx:
                    prompt_text = basename[start_idx:end_idx]
                else:
                    prompt_text = basename[:end_idx] if end_idx > 0 else basename
            
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
    else:
        # text_lang="auto"，遍历所有语言文件夹
        all_langs = [d for d in os.listdir(audio_base_dir) if os.path.isdir(os.path.join(audio_base_dir, d))]
        all_audio_files = []
        emotion_files = []
        
        for lang in all_langs:
            lang_dir = os.path.join(audio_base_dir, lang)
            audio_files = glob.glob(os.path.join(lang_dir, "【*】*.*"))
            if not audio_files:
                audio_files = glob.glob(os.path.join(lang_dir, "*.*"))
            all_audio_files.extend([(f, lang) for f in audio_files])
            if emotion:
                emotion_files.extend([(f, lang) for f in audio_files if f"【{emotion}】" in os.path.basename(f)])
        
        if not all_audio_files:
            return None, None, None
        
        if emotion and emotion_files:
            audio_path, detected_lang = random.choice(emotion_files)
        else:
            audio_path, detected_lang = random.choice(all_audio_files)
        
        txt_path = audio_path.rsplit(".", 1)[0] + ".txt"
        prompt_text = None
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
        else:
            basename = os.path.basename(audio_path)
            start_idx = basename.find("】") + 1
            end_idx = basename.rfind(".")
            if start_idx > 0 and end_idx > start_idx:
                prompt_text = basename[start_idx:end_idx]
            else:
                prompt_text = basename[:end_idx] if end_idx > 0 else basename
        
        return audio_path, prompt_text, detected_lang

config = load_config()
inference = GPTSoVITSInference(
    config.get("bert_base_path"),
    config.get("cnhuhbert_base_path"),
    config.get("device", default_device),
    config.get("is_half", False)
)

inference.load_sovits(config.get("vits_weights_path"), config.get("version", "v2"))
inference.load_gpt(config.get("t2s_weights_path"))

port = args.port
host = args.bind_addr
argv = sys.argv
now_dir = os.getcwd()

APP = FastAPI()

class TTS_Request(BaseModel):
    text: str
    text_lang: str
    ref_audio_path: str
    prompt_lang: str
    prompt_text: Optional[str] = ""
    top_k: Optional[int] = 5
    top_p: Optional[float] = 1
    temperature: Optional[float] = 1
    sample_steps: Optional[int] = 16
    media_type: Optional[str] = "wav"
    streaming_mode: Optional[bool] = False
    threshold: Optional[int] = 30

class TTSRole_Request(BaseModel):
    text: str
    role: str
    text_lang: Optional[str] = "zh"
    ref_audio_path: Optional[str] = None
    prompt_lang: Optional[str] = None
    prompt_text: Optional[str] = None
    emotion: Optional[str] = None
    top_k: Optional[int] = 5
    top_p: Optional[float] = 1
    temperature: Optional[float] = 1
    sample_steps: Optional[int] = 16
    media_type: Optional[str] = "wav"
    streaming_mode: Optional[bool] = False
    threshold: Optional[int] = 30

@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if not weights_path:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        inference.load_gpt(weights_path)
        return JSONResponse(status_code=200, content={"message": "success"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change gpt weight failed", "exception": str(e)})

@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if not weights_path:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        inference.load_sovits(weights_path)
        return JSONResponse(status_code=200, content={"message": "success"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change sovits weight failed", "exception": str(e)})

@APP.get("/tts")
async def tts_get_endpoint(
    text: str,
    text_lang: str = "auto",
    ref_audio_path: str,
    prompt_lang: str,
    prompt_text: str = "",
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    sample_steps: int = 16,
    media_type: str = "wav",
    streaming_mode: bool = False,
    threshold: int = 30
):
    check_res = check_params(text, text_lang, ref_audio_path, prompt_lang, prompt_text, media_type, streaming_mode)
    if check_res is not None:
        return check_res

    try:
        inference.set_prompt_audio(prompt_text, prompt_lang, ref_audio_path)
        version = config.get("version", "v2")

        if streaming_mode:
            print("Streaming_mode...")
            tts_generator = inference.get_tts_wav_stream(text, text_lang, top_k, top_p,
                                                         temperature, sample_steps, version, threshold)
            sample_rate = 24000 if version == "v3" else 32000
            headers = {"sample_rate": str(sample_rate)}
            return StreamingResponse(streaming_generator(tts_generator, "raw"),
                                     headers=headers, media_type=f"audio/{media_type}")
        else:
            sample_rate, audio_data = inference.get_tts_wav(text, text_lang, top_k, top_p,
                                                            temperature, sample_steps, version)
            audio_data = pack_audio(audio_data, sample_rate, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"tts failed", "exception": str(e)})

@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict(exclude_unset=True)
    if "text_lang" in req:
        req["text_lang"] = req["text_lang"].lower()
    if "prompt_lang" in req:
        req["prompt_lang"] = req["prompt_lang"].lower()
    return await tts_get_endpoint(**req)

@APP.get("/ttsrole")
async def ttsrole_get_endpoint(
    text: str,
    role: str,
    text_lang: str = "zh",
    ref_audio_path: Optional[str] = None,
    prompt_lang: Optional[str] = None,
    prompt_text: Optional[str] = None,
    emotion: Optional[str] = None,
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    sample_steps: int = 16,
    media_type: str = "wav",
    streaming_mode: bool = False,
    threshold: int = 30
):
    check_res = check_ttsrole_params(text, role)
    if check_res is not None:
        return check_res

    try:
        role_exists, role_config = load_role_config(role, config)
        if not role_exists:
            return JSONResponse(status_code=400, content={"error": f"Role directory for {role} not found"})

        if ref_audio_path:
            txt_path = ref_audio_path.rsplit(".", 1)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
            else:
                basename = os.path.basename(ref_audio_path)
                if "【" in basename and "】" in basename:
                    start_idx = basename.find("】") + 1
                    end_idx = basename.rfind(".")
                    prompt_text = basename[start_idx:end_idx] if start_idx > 0 and end_idx > start_idx else basename[:end_idx] if end_idx > 0 else basename
                else:
                    end_idx = basename.rfind(".")
                    prompt_text = basename[:end_idx] if end_idx > 0 else basename
            inference.set_prompt_audio(prompt_text, prompt_lang, ref_audio_path)
        else:
            ref_audio_path, prompt_text, detected_lang = select_ref_audio(role, text_lang, emotion)
            if not ref_audio_path:
                return JSONResponse(status_code=400, content={"error": "No suitable reference audio found for role"})
            prompt_lang = detected_lang
            inference.set_prompt_audio(prompt_text, prompt_lang, ref_audio_path)

        version = role_config.get("version", "v2")
        if streaming_mode:
            print("Streaming_mode...")
            tts_generator = inference.get_tts_wav_stream(text, text_lang, top_k, top_p,
                                                         temperature, sample_steps, version, threshold)
            sample_rate = 24000 if version == "v3" else 32000
            headers = {"sample_rate": str(sample_rate)}
            return StreamingResponse(streaming_generator(tts_generator, "raw"),
                                     headers=headers, media_type=f"audio/{media_type}")
        else:
            sample_rate, audio_data = inference.get_tts_wav(text, text_lang, top_k, top_p,
                                                            temperature, sample_steps, version)
            audio_data = pack_audio(audio_data, sample_rate, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"ttsrole failed", "exception": str(e)})

@APP.post("/ttsrole")
async def ttsrole_post_endpoint(request: TTSRole_Request):
    req = request.dict(exclude_unset=True)
    if "text_lang" in req:
        req["text_lang"] = req["text_lang"].lower()
    if "prompt_lang" in req and req["prompt_lang"]:
        req["prompt_lang"] = req["prompt_lang"].lower()
    return await ttsrole_get_endpoint(**req)

def streaming_generator(tts_generator: Generator, media_type: str):
    for sr, chunk in tts_generator:
        yield pack_audio(chunk, sr, media_type).getvalue()

if __name__ == "__main__":
    try:
        if host == 'None':  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
