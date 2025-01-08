"""
     ██╗ ██████╗ ██╗   ██╗██╗    ███████╗██████╗  ██████╗ ██╗   ██╗████████╗
     ██║██╔═══██╗██║   ██║██║    ██╔════╝██╔══██╗██╔═══██╗██║   ██║╚══██╔══╝
     ██║██║   ██║██║   ██║██║    ███████╗██████╔╝██║   ██║██║   ██║   ██║
██   ██║██║   ██║╚██╗ ██╔╝██║    ╚════██║██╔═══╝ ██║   ██║██║   ██║   ██║
╚█████╔╝╚██████╔╝ ╚████╔╝ ██║    ███████║██║     ╚██████╔╝╚██████╔╝   ██║
 ╚════╝  ╚═════╝   ╚═══╝  ╚═╝    ╚══════╝╚═╝      ╚═════╝  ╚═════╝    ╚═╝

                       SPOUT support for ComfyUI
                http://www.github.com/amorano/Jovi_Spout
"""

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
__author__ = """Alexander G. Morano"""
__email__ = "amorano@gmail.com"
__version__ = "1.0.4"

import os
import sys
import json
import inspect
import importlib
from pathlib import Path
from types import ModuleType
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

JOV_PACKAGE = "JOV_SPOUT"

# ==============================================================================
# === CORE NODES ===
# ==============================================================================

class JOVBaseNode:
    NOT_IDEMPOTENT = True
    CATEGORY = f"{JOV_PACKAGE} 📺"
    RETURN_TYPES = ()
    FUNCTION = "run"

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float('nan')

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

# ==============================================================================
# === TYPE ===
# ==============================================================================

class AnyType(str):
    """AnyType input wildcard trick taken from pythongossss's:

    https://github.com/pythongosssss/ComfyUI-Custom-Scripts
    """
    def __ne__(self, __value: object) -> bool:
        return False

JOV_TYPE_ANY = AnyType("*")

# ==============================================================================
# === NODE LOADER ===
# ==============================================================================

def load_module(name: str) -> None|ModuleType:
    module = inspect.getmodule(inspect.stack()[0][0]).__name__
    try:
        route = str(name).replace("\\", "/")
        route = route.split(f"{module}/core/")[1]
        route = route.split('.')[0].replace('/', '.')
    except Exception as e:
        logger.warning(f"module failed {name}")
        logger.warning(str(e))
        return

    try:
        module = f"{module}.core.{route}"
        module = importlib.import_module(module)
    except Exception as e:
        logger.warning(f"module failed {module}")
        logger.warning(str(e))
        return

    return module

def loader():
    global NODE_DISPLAY_NAME_MAPPINGS, NODE_CLASS_MAPPINGS
    NODE_LIST_MAP = {}

    for fname in ROOT.glob('core/**/*.py'):
        if fname.stem.startswith('_'):
            continue

        if (module := load_module(fname)) is None:
            continue

        classes = inspect.getmembers(module, inspect.isclass)
        for class_name, class_object in classes:
            if not class_name.endswith('BaseNode') and hasattr(class_object, 'NAME') and hasattr(class_object, 'CATEGORY'):
                name = f"{class_object.NAME} ({JOV_PACKAGE})"
                NODE_DISPLAY_NAME_MAPPINGS[name] = name
                NODE_CLASS_MAPPINGS[name] = class_object
                desc = class_object.DESCRIPTION if hasattr(class_object, 'DESCRIPTION') else name
                NODE_LIST_MAP[name] = desc.split('.')[0].strip('\n')

    NODE_CLASS_MAPPINGS = {x[0] : x[1] for x in sorted(NODE_CLASS_MAPPINGS.items(),
                                                            key=lambda item: getattr(item[1], 'SORT', 0))}

    keys = NODE_CLASS_MAPPINGS.keys()
    for name in keys:
        logger.debug(f"✅ {name}")
    logger.info(f"{len(keys)} nodes loaded")

    # only do the list on local runs...
    if JOV_INTERNAL:
        with open(str(ROOT) + "/node_list.json", "w", encoding="utf-8") as f:
            json.dump(NODE_LIST_MAP, f, sort_keys=True, indent=4 )

loader()
