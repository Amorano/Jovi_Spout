"""
Jovi_Spout - http://www.github.com/Amorano/Jovi_Spout
Core
"""

from .. import JOVBaseNode

# ==============================================================================
# === CORE NODES ===
# ==============================================================================

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

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

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
