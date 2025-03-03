import logging
import re
from pathlib import Path
import os
from typing import Optional

# 静音 jieba
import jieba
jieba.setLogLevel(logging.CRITICAL)

# 配置 fast_langdetect
import fast_langdetect
from fast_langdetect import detect

# 设置自定义缓存路径（仅在支持的情况下使用）
try:
    # 检查是否支持 CACHE_DIRECTORY 属性
    if hasattr(fast_langdetect.ft_detect.infer, 'CACHE_DIRECTORY'):
        fast_langdetect.ft_detect.infer.CACHE_DIRECTORY = Path(__file__).parent.parent.parent / "pretrained_models" / "fast_langdetect"
except AttributeError:
    # 如果 ft_detect 不存在，使用环境变量或默认路径
    os.environ['FASTTEXT_MODEL_PATH'] = str(Path(__file__).parent.parent.parent / "pretrained_models" / "fast_langdetect" / "lid.176.bin")

# Windows 下模型加载的兼容性处理
def load_fasttext_model(model_path: Path, download_url: Optional[str] = None, proxy: Optional[str] = None):
    """
    Load a FastText model with fallback for Windows compatibility.
    """
    if not model_path.exists() and download_url:
        logging.warning(f"Model not found at {model_path}, attempting to download from {download_url}")
        fast_langdetect.ft_detect.infer.download_model(download_url, model_path, proxy)
        if not model_path.exists():
            raise ValueError(f"Failed to download model to {model_path}")

    try:
        # 直接加载模型，路径使用 str 格式
        model = fast_langdetect.ft_detect.infer.fasttext.load_model(str(model_path))
        return model
    except Exception as e:
        logging.error(f"Failed to load FastText model from {model_path}: {e}")
        raise

# 如果是 Windows，覆盖默认加载函数
if os.name == 'nt' and hasattr(fast_langdetect.ft_detect.infer, 'load_fasttext_model'):
    fast_langdetect.ft_detect.infer.load_fasttext_model = load_fasttext_model

from split_lang import LangSplitter

# 判断是否全为英文字符
def full_en(text):
    pattern = r'^[A-Za-z0-9\s\u0020-\u007E\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]+$'
    return bool(re.match(pattern, text))

# 提取 CJK 字符
def full_cjk(text):
    cjk_ranges = [
        (0x4E00, 0x9FFF), (0x3400, 0x4DB5), (0x20000, 0x2A6DD), (0x2A700, 0x2B73F),
        (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0x2CEB0, 0x2EBEF), (0x30000, 0x3134A),
        (0x31350, 0x323AF), (0x2EBF0, 0x2EE5D),
    ]
    pattern = r'[0-9、-〜。！？.!?… ]'
    cjk_text = "".join(
        char for char in text
        if any(start <= ord(char) <= end for start, end in cjk_ranges) or re.match(pattern, char)
    )
    return cjk_text

# 分割日文或韩文片段
def split_jako(tag_lang, item):
    if tag_lang == "ja":
        pattern = r"([\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]+(?:[0-9、-〜。！？.!?… ]+[\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]*)*)"
    else:  # ko
        pattern = r"([\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]+(?:[0-9、-〜。！？.!?… ]+[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]*)*)"

    lang_list = []
    tag = 0
    for match in re.finditer(pattern, item['text']):
        if match.start() > tag:
            lang_list.append({'lang': item['lang'], 'text': item['text'][tag:match.start()]})
        lang_list.append({'lang': tag_lang, 'text': match.group(0)})
        tag = match.end()
    if tag < len(item['text']):
        lang_list.append({'lang': item['lang'], 'text': item['text'][tag:]})
    return lang_list

# 合并相同语言的片段
def merge_lang(lang_list, item):
    if lang_list and item['lang'] == lang_list[-1]['lang']:
        lang_list[-1]['text'] += item['text']
    else:
        lang_list.append(item)
    return lang_list

class LangSegmenter:
    DEFAULT_LANG_MAP = {
        "zh": "zh", "yue": "zh", "wuu": "zh", "zh-cn": "zh", "zh-tw": "x",
        "ko": "ko", "ja": "ja", "en": "en",
    }

    @staticmethod
    def getTexts(text):
        lang_splitter = LangSplitter(lang_map=LangSegmenter.DEFAULT_LANG_MAP)
        substr = lang_splitter.split_by_lang(text=text)
        lang_list = []

        for item in substr:
            dict_item = {'lang': item.lang, 'text': item.text}

            # 处理短英文被误识别的情况
            if full_en(dict_item['text']):
                dict_item['lang'] = 'en'
                lang_list = merge_lang(lang_list, dict_item)
                continue

            # 处理非日语中的日文片段
            ja_list = split_jako('ja', dict_item) if dict_item['lang'] != 'ja' else [dict_item]

            # 处理非韩语中的韩文片段
            temp_list = []
            for ja_item in ja_list:
                ko_list = split_jako('ko', ja_item) if ja_item['lang'] != 'ko' else [ja_item]
                temp_list.extend(ko_list)

            # 处理结果
            for temp_item in temp_list:
                if temp_item['lang'] == 'x':
                    cjk_text = full_cjk(temp_item['text'])
                    if cjk_text:
                        temp_item = {'lang': 'zh', 'text': cjk_text}
                lang_list = merge_lang(lang_list, temp_item)

        return lang_list

if __name__ == "__main__":
    text1 = "MyGO?,你也喜欢まいご吗？"
    print(LangSegmenter.getTexts(text1))

    text2 = "ねえ、知ってる？最近、僕は天文学を勉強してるんだ。君の瞳が星空みたいにキラキラしてるからさ。"
    print(LangSegmenter.getTexts(text2))
