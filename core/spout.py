"""
Jovi_Spout - Device -- SPOUT
"""

import time
import array
import threading
from enum import Enum
from typing import Tuple
from itertools import repeat

import cv2
import torch
import SpoutGL
import numpy as np
from OpenGL import GL

from comfy.utils import ProgressBar

from cozy_comfyui import \
    logger, \
    deep_merge

from cozy_comfyui.node import \
    CozyBaseNode, CozyImageNode

from cozy_comfyui.image.convert import \
    cv_to_tensor_full, tensor_to_cv, image_convert

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumInterpolation(Enum):
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    LANCZOS4 = cv2.INTER_LANCZOS4
    LINEAR_EXACT = cv2.INTER_LINEAR_EXACT
    NEAREST_EXACT = cv2.INTER_NEAREST_EXACT

# ==============================================================================
# === CLASS ===
# ==============================================================================

class SpoutReaderNode(CozyImageNode):
    NAME = "SPOUT READER"
    SORT = 50
    DESCRIPTION = """
Capture frames from Spout streams. It supports batch processing, allowing multiple frames to be captured simultaneously. The node provides options for configuring the source and number of frames to gather. The captured frames are returned as tensors, enabling further processing downstream.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "required": {
                'url': ("STRING", {"default": "Spout Graphics Sender", "dynamicPrompts": False, "tooltip": "source of the Spout stream to capture."}),
                'width': ("INT", {"default": 512, "min": 32, "max": 2048, "tooltip":"width of image after reading from Spout stream."}),
                'height': ("INT", {"default": 512, "min": 32, "max": 2048, "tooltip":"height of image after reading from Spout stream."}),
                'fps': ("INT", {"default": 30, "min": 1, "max": 60, "tooltip":"frames per second to capture."}),
                'sample': (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name, "tooltip":"If the images is smaller or larger, the interpolation method to rescale the stream image."}),
                'batch': ("INT", {"default": 1, "min": 1, "max": 1024, "tooltip": "collect multiple frames at once."}),
            }
        })
        return d

    def run(self, **kw) -> Tuple[torch.Tensor, ...]:
        delta = 1. / kw['fps'][0]
        count = kw['batch'][0]
        sample = kw['sample'][0]
        width = kw['width'][0]
        height = kw['height'][0]
        url = kw['url'][0]
        sample = EnumInterpolation[sample].value
        blank = np.zeros((width, height, 4), dtype=np.uint8)
        frames = [blank] * count
        width = height = 0
        buffer = None
        idx = 0
        pbar = ProgressBar(count)
        with SpoutGL.SpoutReceiver() as receiver:
            receiver.setReceiverName(url)
            while idx <= count:
                waste = time.perf_counter() + delta
                receiver.waitFrameSync(url, 0)
                if buffer is None or receiver.isUpdated():
                    width = receiver.getSenderWidth()
                    height = receiver.getSenderHeight()
                    buffer = array.array('B', repeat(0, width * height * 4))
                    # logger.debug(f"{width} x {height}")
                    # logger.debug("changed")

                result = receiver.receiveImage(buffer, GL.GL_RGBA, False, 0)
                if result:
                    if SpoutGL.helpers.isBufferEmpty(buffer):
                        logger.debug("empty")
                        continue

                    if idx > 0:
                        frames[idx-1] = np.asarray(buffer, dtype=np.uint8).reshape((height, width, 4))
                    waste = max(waste - time.perf_counter(), 0)
                    # logger.debug(f"{idx-1} - {waste}")
                    idx += 1
                    if count > 1:
                        time.sleep(waste)
                buffer = None
                receiver.setFrameSync(url)
                pbar.update_absolute(idx)

        frames = [cv_to_tensor_full(cv2.resize(i, (width, height),
                                            interpolation=sample)) for i in frames]
        return [torch.stack(i) for i in zip(*frames)]

class SpoutWriterNode(CozyBaseNode):
    NAME = "SPOUT WRITER"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    SORT = 90
    DESCRIPTION = """
Sends frame(s) to a specified Spout receiver application for real-time video sharing. Accepts tensors representing images. The node continuously streams frames to the specified Spout host, enabling real-time visualization or integration with other applications that support Spout.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "required": {
                'image': ("IMAGE", {"default": None, "tooltip": "RGBA, RGB or Grayscale image"}),
                'url': ("STRING", {"default": "Spout Sender", "tooltip": "source of the Spout stream to send."}),
                'fps': ("INT", {"default": 30, "min": 1, "max": 60, "tooltip":"frames per second to capture."}),
            }
        })
        return d

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__frame = None
        self.__host = ''
        self.__delay = 0.05
        self.__sender = SpoutGL.SpoutSender()
        self.__thread_server = threading.Thread(target=self.__server, daemon=True)
        self.__thread_server.start()

    def __server(self) -> None:
        while 1:
            try:
                h, w = self.__frame.shape[:2]
                buffer = np.ascontiguousarray(self.__frame, dtype=np.uint8)
                self.__sender.sendImage(buffer, w, h, GL.GL_RGBA, False, 0)
                self.__sender.setFrameSync(self.__host)
                # logger.debug(self.__host)
            except AttributeError as e:
                pass
            finally:
                time.sleep(self.__delay)

    def run(self, **kw) -> None:

        if (url := kw['url'][0]) != self.__host:
            if self.__sender is not None:
                self.__sender.releaseSender()
            self.__sender = SpoutGL.SpoutSender()
            self.__host = url
            self.__sender.setSenderName(self.__host)

        fps = kw['fps'][0]
        self.__delay = 1. / min(60, max(1, fps))

        images = kw['image']
        pbar = ProgressBar(len(images))
        for idx, img in enumerate(images):
            img = tensor_to_cv(img)
            self.__frame = image_convert(img, 4)
            pbar.update_absolute(idx)
        return ()
