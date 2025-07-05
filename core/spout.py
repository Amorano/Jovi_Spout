""" Jovi_Spout - Device -- SPOUT """

import time
import array
import threading
from itertools import repeat

import SpoutGL
import numpy as np
from OpenGL import GL

from comfy.utils import ProgressBar

from cozy_comfyui import \
    logger, \
    IMAGE_SIZE_MIN, IMAGE_SIZE_MAX, IMAGE_SIZE_DEFAULT, \
    RGBAMaskType, EnumConvertType, \
    deep_merge, parse_param

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.node import \
    COZY_TYPE_IMAGE, \
    CozyBaseNode, CozyImageNode

from cozy_comfyui.image.convert import \
    cv_to_tensor_full, tensor_to_cv

from cozy_comfyui.image.compose import \
    EnumInterpolation, \
    image_resize

from cozy_comfyui.image.misc import \
    image_stack

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
                Lexicon.URL: ("STRING", {
                    "default": "Spout Graphics Sender", "dynamicPrompts": False,
                    "tooltip": "Source of the Spout stream to capture."}),
                Lexicon.WIDTH: ("INT", {
                    "default": IMAGE_SIZE_DEFAULT, "min": IMAGE_SIZE_MIN, "max": IMAGE_SIZE_MAX,
                    "tooltip":"Width of image after reading from Spout stream."}),
                Lexicon.HEIGHT: ("INT", {
                    "default": IMAGE_SIZE_DEFAULT, "min": IMAGE_SIZE_MIN, "max": IMAGE_SIZE_MAX,
                    "tooltip":"Height of image after reading from Spout stream."}),
                Lexicon.FPS: ("INT", {
                    "default": 30, "min": 1, "max": 60}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {
                    "default": EnumInterpolation.LANCZOS4.name,
                    "tooltip":"If the images is smaller or larger, the interpolation method to rescale the stream image."}),
                Lexicon.BATCH: ("INT", {
                    "default": 1, "min": 1, "max": 3600,
                    "tooltip": "Collect multiple frames at once."}),
                Lexicon.TIMEOUT: ("INT", {
                    "default": 5, "min": 2, "max": 10,}),
            }
        })
        return Lexicon._parse(d)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float('nan')

    def run(self, **kw) -> RGBAMaskType:
        delta = 1. / kw[Lexicon.FPS][0]
        count = kw[Lexicon.BATCH][0]
        sample = kw[Lexicon.SAMPLE][0]
        width = kw[Lexicon.WIDTH][0]
        height = kw[Lexicon.HEIGHT][0]
        timeout = kw[Lexicon.TIMEOUT][0]
        url = kw[Lexicon.URL][0]
        blank = np.zeros((width, height, 4), dtype=np.uint8)
        frames = [blank] * count
        w = h = 0
        buffer = None
        idx = 0
        pbar = ProgressBar(count)
        with SpoutGL.SpoutReceiver() as receiver:
            receiver.setReceiverName(url)
            timeout_spent = 0
            while idx <= count:
                timeout_counter = time.perf_counter()
                waste = timeout_counter + delta
                receiver.waitFrameSync(url, 0)
                if buffer is None or receiver.isUpdated():
                    w = receiver.getSenderWidth()
                    h = receiver.getSenderHeight()
                    buffer = array.array('B', repeat(0, w * h * 4))

                result = receiver.receiveImage(buffer, GL.GL_RGBA, False, 0)
                if result:
                    if SpoutGL.helpers.isBufferEmpty(buffer):
                        # logger.debug("empty")
                        continue

                    if idx > 0:
                        frames[idx-1] = np.asarray(buffer, dtype=np.uint8).reshape((h, w, 4))
                    waste = max(waste - time.perf_counter(), 0)
                    idx += 1
                    if count > 1:
                        time.sleep(waste)
                    buffer = None
                    receiver.setFrameSync(url)
                    pbar.update_absolute(idx)
                    timeout_spent = 0
                else:
                    timeout_spent += (time.perf_counter() - timeout_counter)
                    if timeout_spent >= timeout:
                        logger.error(f"timeout reading SPOUT stream {url}")
                        return ()

        sample = EnumInterpolation[sample].value
        frames = [cv_to_tensor_full(image_resize(i, width, height, sample)) for i in frames]
        return image_stack(frames)

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
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {
                    "default": None}),
                Lexicon.URL: ("STRING", {
                    "default": "Spout Sender",
                    "tooltip": "source of the Spout stream to send."}),
                Lexicon.FPS: ("INT", {
                    "default": 30, "min": 1, "max": 60}),
            }
        })
        return Lexicon._parse(d)

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

        if (url := kw[Lexicon.URL][0]) != self.__host:
            if self.__sender is not None:
                self.__sender.releaseSender()
            self.__sender = SpoutGL.SpoutSender()
            self.__host = url
            self.__sender.setSenderName(self.__host)

        fps = parse_param(kw, Lexicon.FPS, EnumConvertType.FLOAT, 30, 1, 60)[0]
        self.__delay = 1. / fps

        images = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        pbar = ProgressBar(len(images))
        for idx, img in enumerate(images):
            self.__frame = tensor_to_cv(img, chan=4)
            pbar.update_absolute(idx)
        return ()
