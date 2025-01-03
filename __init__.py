"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

                           SPOUT support for ComfyUI
                    http://www.github.com/amorano/Jovi_Spout

@title: Jovi_Spout
@author: Alexander G. Morano
@category: Stream
@reference: https://github.com/Amorano/Jovi_Spout
@tags: Stream, Spout, Media
@description: SPOUT support for ComfyUI.
@node list:
    SpoutReaderNode, SpoutWriterNode
@version: 1.0.0
"""

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
__author__ = """Alexander G. Morano"""
__email__ = "amorano@gmail.com"
__version__ = "1.0.0"

import os
import sys
import json
import inspect
import importlib
from pathlib import Path
from typing import Any

from loguru import logger

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

ROOT = Path(__file__).resolve().parent
ROOT_COMFY = ROOT.parent.parent

JOV_WEB = ROOT / 'web'

JOV_LOG_LEVEL = os.getenv("JOV_LOG_LEVEL", "INFO")
logger.configure(handlers=[{"sink": sys.stdout, "level": JOV_LOG_LEVEL}])

JOV_INTERNAL = os.getenv("JOV_INTERNAL", 'false').strip().lower() in ('true', '1', 't')

# ==============================================================================
# === THERE CAN BE ONLY ONE ===
# ==============================================================================

class Singleton(type):
    _instances = {}

    def __call__(cls, *arg, **kw) -> Any:
        # If the instance does not exist, create and store it
        if cls not in cls._instances:
            instance = super().__call__(*arg, **kw)
            cls._instances[cls] = instance
        return cls._instances[cls]

# ==============================================================================
# === CORE NODES ===
# ==============================================================================

class JOVBaseNode:
    NOT_IDEMPOTENT = True
    CATEGORY = f"JOVI_SPOUT 📺"
    RETURN_TYPES = ()
    FUNCTION = "run"

    @classmethod
    def VALIDATE_INPUTS(cls, *arg, **kw) -> bool:
        return True

    @classmethod
    def INPUT_TYPES(cls, prompt:bool=False, extra_png:bool=False, dynprompt:bool=False) -> dict:
        data = {
            "required": {},
            "hidden": {
                "ident": "UNIQUE_ID"
            }
        }
        if prompt:
            data["hidden"]["prompt"] = "PROMPT"
        if extra_png:
            data["hidden"]["extra_pnginfo"] = "EXTRA_PNGINFO"

        if dynprompt:
            data["hidden"]["dynprompt"] = "DYNPROMPT"
        return data

class JOVImageNode(JOVBaseNode):
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("RGBA", "RGB", "MASK")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "outputs": {
                0: ("IMAGE", {"tooltip":"Full channel [RGBA] image. If there is an alpha, the image will be masked out with it when using this output."}),
                1: ("IMAGE", {"tooltip":"Three channel [RGB] image. There will be no alpha."}),
                2: ("MASK", {"tooltip":"Single channel mask output."}),
            }
        })
        return d

class AnyType(str):
    """AnyType input wildcard trick taken from pythongossss's:

    https://github.com/pythongosssss/ComfyUI-Custom-Scripts
    """
    def __ne__(self, __value: object) -> bool:
        return False

JOV_TYPE_ANY = AnyType("*")
JOV_TYPE_IMAGE = JOV_TYPE_ANY

def deep_merge(d1: dict, d2: dict) -> dict:
    """
    Deep merge multiple dictionaries recursively.

    Args:
        *dicts: Variable number of dictionaries to be merged.

    Returns:
        dict: Merged dictionary.
    """
    for key in d2:
        if key in d1:
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                deep_merge(d1[key], d2[key])
            else:
                d1[key] = d2[key]
        else:
            d1[key] = d2[key]
    return d1

# ==============================================================================
# === NODE LOADER ===
# ==============================================================================

def loader():
    node_count = 0
    NODE_LIST_MAP = {}

    global NODE_DISPLAY_NAME_MAPPINGS, NODE_CLASS_MAPPINGS

    for fname in ROOT.glob('core/**/*.py'):
        if fname.stem.startswith('_'):
            continue

        try:
            route = str(fname).replace("\\", "/").split("Jovi_Spout/core/")[1]
            route = route.split('.')[0].replace('/', '.')
            module = f"Jovi_Spout.core.{route}"
            module = importlib.import_module(module)
        except Exception as e:
            logger.warning(f"module failed {fname}")
            logger.warning(str(e))
            continue

        classes = inspect.getmembers(module, inspect.isclass)
        for class_name, class_object in classes:
            # assume both attrs are good enough....
            if not class_name.endswith('BaseNode') and hasattr(class_object, 'NAME') and hasattr(class_object, 'CATEGORY'):
                name = class_object.NAME
                NODE_DISPLAY_NAME_MAPPINGS[name] = name
                NODE_CLASS_MAPPINGS[name] = class_object
                desc = class_object.DESCRIPTION if hasattr(class_object, 'DESCRIPTION') else name
                NODE_LIST_MAP[name] = desc.split('.')[0].strip('\n')
                node_count += 1

        logger.info(f"✅ {module.__name__}")
    logger.info(f"{node_count} nodes loaded")

    # only do the list on local runs...
    if JOV_INTERNAL:
        with open(str(ROOT) + "/node_list.json", "w", encoding="utf-8") as f:
            json.dump(NODE_LIST_MAP, f, sort_keys=True, indent=4 )

loader()
