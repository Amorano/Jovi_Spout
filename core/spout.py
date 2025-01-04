"""
Jovi_Spout - http://www.github.com/amorano/Jovi_Spout
Device -- SPOUT
"""

import time
import array
import threading
from enum import Enum
from typing import Tuple, Union
from itertools import repeat

import cv2
import torch
import SpoutGL
import numpy as np
from OpenGL import GL
from loguru import logger

from comfy.utils import ProgressBar

from Jovi_Spout import JOV_TYPE_IMAGE, JOVBaseNode, JOVImageNode, deep_merge

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

TYPE_iRGB  = Tuple[int, int, int]
TYPE_iRGBA = Tuple[int, int, int, int]
TYPE_fRGB  = Tuple[float, float, float]
TYPE_fRGBA = Tuple[float, float, float, float]

TYPE_PIXEL = Union[int, float, TYPE_iRGB, TYPE_iRGBA, TYPE_fRGB, TYPE_fRGBA]
TYPE_IMAGE = Union[np.ndarray, torch.Tensor]

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
# === SUPPORT ===
# ==============================================================================

def image_mask(image: TYPE_IMAGE, color: TYPE_PIXEL = 255) -> TYPE_IMAGE:
    """Create a mask from the image, preserving transparency.

    Args:
        image (TYPE_IMAGE): Input image, assumed to be 2D or 3D (with or without alpha channel).
        color (TYPE_PIXEL): Value to fill the mask (default is 255).

    Returns:
        TYPE_IMAGE: Mask of the image, either the alpha channel or a full mask of the given color.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        return image[..., 3]

    h, w = image.shape[:2]
    return np.ones((h, w), dtype=np.uint8) * color

def image_matte(image: TYPE_IMAGE, color: TYPE_iRGBA=(0, 0, 0, 255), width: int=None, height: int=None) -> TYPE_IMAGE:
    """
    Puts an RGB(A) image atop a colored matte expanding or clipping the image if requested.

    Args:
        image (TYPE_IMAGE): The input RGBA image.
        color (TYPE_iRGBA): The color of the matte as a tuple (R, G, B, A).
        width (int, optional): The width of the matte. Defaults to the image width.
        height (int, optional): The height of the matte. Defaults to the image height.

    Returns:
        TYPE_IMAGE: Composited RGBA image on a matte with original alpha channel.
    """

    # Determine the dimensions of the image and the matte
    image_height, image_width = image.shape[:2]
    width = width or image_width
    height = height or image_height

    # Create a solid matte with the specified color
    matte = np.full((height, width, 4), color, dtype=image.dtype)

    # Calculate the center position for the image on the matte
    x_offset = (width - image_width) // 2
    y_offset = (height - image_height) // 2

    # Extract the alpha channel from the image if it's RGBA
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3] / 255.0

        # Blend the RGB channels using the alpha mask
        for c in range(3):  # Iterate over RGB channels
            matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] = \
                (1 - alpha) * matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] + \
                alpha * image[:, :, c]

        # Set the alpha channel to the image's alpha channel
        matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, 3] = image[:, :, 3]
    else:
        # Handle non-RGBA images (just copy the image onto the matte)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=-1)
        matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, :3] = image[:, :, :3]

    return matte

def image_convert(image: TYPE_IMAGE, channels: int, width: int=None, height: int=None,
                  matte: Tuple[int, ...]=(0, 0, 0, 255)) -> TYPE_IMAGE:
    """Force image format to a specific number of channels.
    Args:
        image (TYPE_IMAGE): Input image.
        channels (int): Desired number of channels (1, 3, or 4).
        width (int): Desired width. `None` means leave unchanged.
        height (int): Desired height. `None` means leave unchanged.
        matte (tuple): RGBA color to use as background color for transparent areas.
    Returns:
        TYPE_IMAGE: Image with the specified number of channels.
    """
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    if (cc := image.shape[2]) != channels:
        if   cc == 1 and channels == 3:
            image = np.repeat(image, 3, axis=2)
        elif cc == 1 and channels == 4:
            rgb = np.repeat(image, 3, axis=2)
            alpha = np.full(image.shape[:2] + (1,), matte[3], dtype=image.dtype)
            image = np.concatenate([rgb, alpha], axis=2)
        elif cc == 3 and channels == 1:
            image = np.mean(image, axis=2, keepdims=True).astype(image.dtype)
        elif cc == 3 and channels == 4:
            alpha = np.full(image.shape[:2] + (1,), matte[3], dtype=image.dtype)
            image = np.concatenate([image, alpha], axis=2)
        elif cc == 4 and channels == 1:
            rgb = image[..., :3]
            alpha = image[..., 3:4] / 255.0
            image = (np.mean(rgb, axis=2, keepdims=True) * alpha).astype(image.dtype)
        elif cc == 4 and channels == 3:
            image = image[..., :3]

    # Resize if width or height is specified
    h, w = image.shape[:2]
    new_width = width if width is not None else w
    new_height = height if height is not None else h
    if (new_width, new_height) != (w, h):
        # Create a new canvas with the specified dimensions and matte color
        new_image = np.full((new_height, new_width, channels), matte[:channels], dtype=image.dtype)

        # Calculate the region of the original image to copy over
        src_x1 = max(0, (w - new_width) // 2) if new_width < w else 0
        src_y1 = max(0, (h - new_height) // 2) if new_height < h else 0
        src_x2 = src_x1 + min(w, new_width)
        src_y2 = src_y1 + min(h, new_height)

        # Calculate the region of the new image to paste onto
        dst_x1 = max(0, (new_width - w) // 2) if new_width > w else 0
        dst_y1 = max(0, (new_height - h) // 2) if new_height > h else 0
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        # Place the original image onto the new image
        new_image[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        image = new_image

    return image

def tensor2cv(tensor: torch.Tensor, invert_mask:bool=True) -> TYPE_IMAGE:
    """Convert a torch Tensor to a numpy ndarray."""
    if tensor.ndim > 3:
        raise Exception("Tensor is batch of tensors")

    if tensor.ndim < 3:
        tensor = tensor.unsqueeze(-1)

    if tensor.shape[2] == 1 and invert_mask:
        tensor = 1. - tensor

    tensor = tensor.cpu().numpy()
    return np.clip(255.0 * tensor, 0, 255).astype(np.uint8)

def cv2tensor_full(image: TYPE_IMAGE, matte:TYPE_PIXEL=(0,0,0,255)) -> Tuple[torch.Tensor, ...]:

    rgba = image_convert(image, 4)
    rgb = image_matte(rgba, matte)[...,:3]
    mask = image_mask(image)
    rgba = torch.from_numpy(rgba.astype(np.float32) / 255.0)
    rgb = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    mask = torch.from_numpy(mask.astype(np.float32) / 255.0)
    return rgba, rgb, mask

# ==============================================================================
# === COMFYUI NODE ===
# ==============================================================================

class SpoutReaderNode(JOVImageNode):
    NAME = "SPOUT READER (JOV_SP) 📺"
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
        delta = 1. / kw['fps']
        count = kw['batch']
        sample = EnumInterpolation[kw['sample']].value
        blank = np.zeros((kw['width'], kw['height'], 4), dtype=np.uint8)
        frames = [blank] * count
        width = height = 0
        buffer = None
        idx = 0
        pbar = ProgressBar(count)
        with SpoutGL.SpoutReceiver() as receiver:
            receiver.setReceiverName(kw['url'])
            while idx <= count:
                waste = time.perf_counter() + delta
                receiver.waitFrameSync(kw['url'], 0)
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
                receiver.setFrameSync(kw['url'])
                pbar.update_absolute(idx)

        frames = [cv2tensor_full(cv2.resize(i, (kw['width'], kw['height']),
                                            interpolation=sample)) for i in frames]
        return [torch.stack(i) for i in zip(*frames)]

class SpoutWriterNode(JOVBaseNode):
    NAME = "SPOUT WRITER (JOV_SP) 🎥"
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
                'image': (JOV_TYPE_IMAGE, {}),
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
                self.__sender.sendImage(self.__frame, w, h, GL.GL_RGBA, False, 0)
                self.__sender.setFrameSync(self.__host)
                # logger.debug(self.__host)
            except AttributeError as e:
                pass
            finally:
                time.sleep(self.__delay)

    def run(self, **kw) -> None:

        if kw['url'] != self.__host:
            if self.__sender is not None:
                self.__sender.releaseSender()
            self.__sender = SpoutGL.SpoutSender()
            self.__host = kw['url']
            self.__sender.setSenderName(self.__host)

        self.__delay = 1. / min(60, max(1, kw['fps']))

        images = kw['image']
        pbar = ProgressBar(len(images))
        for idx, img in enumerate(images):
            img = tensor2cv(img)
            self.__frame = image_convert(img, 4)
            pbar.update_absolute(idx)
        return ()
