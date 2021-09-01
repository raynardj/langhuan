from datetime import datetime
from flask import request
from typing import Union
import json
from pathlib import Path
import regex


def now_str(): return datetime.now().strftime("%y%m%d_%H%M%S")


def now_specific():
    return datetime.now().strftime("%y-%m-%d_%H:%M:%S")


def cleanup_tags(x: str) -> str:
    """
    remove the string that will break the frontend
    x: str, input string
    """
    return x.replace("<", "˂").replace(">", "˃")


def arg_by_key(key: str) -> Union[str, int, float]:
    """
    get a value either by GET or POST method
    with api function
    """
    if request.method == "POST":
        data = json.loads(request.data)
        rt = data[key]
    elif request.method == "GET":
        rt = request.args.get(key)
    else:
        raise RuntimeError(
            "method has to be GET or POST")
    return rt


def get_root() -> Path:
    return Path(__file__).parent.absolute()


def findall_word_position(text: str, word: str) -> list:
    """
    find all the position of word in text
    """
    text = text.lower()
    word = word.lower()
    return [m.start() for m in regex.finditer(word, text)]
