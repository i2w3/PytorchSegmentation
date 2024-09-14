from typing import List

import numpy as np

class MaskOverflow(Exception):
    """掩码索引应该在[0, n_classes)中"""

class MaskDimOverflow(Exception):
    """掩码图像应该为2维"""

class PaletteOverflow(Exception):
    """调色板索引不够"""

def PaletteArray(array:np.ndarray, palette:List[int], num_classes:int) -> np.ndarray:
    assert isinstance(array, np.ndarray), f"Wrong data type of input array"
    if (array.min() < 0) or (array.max() >= num_classes):
        raise MaskOverflow
    if num_classes * 3 > len(palette):
        raise PaletteOverflow
    if array.ndim != 2:
        raise MaskDimOverflow

    images = np.zeros((*array.shape, 3), dtype=np.uint8)
    for i in range(num_classes):
        images[array == i] = palette[i*3:i*3+1]
    return images
