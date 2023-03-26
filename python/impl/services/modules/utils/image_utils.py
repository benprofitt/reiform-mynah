from typing import Tuple
from PIL import Image # type: ignore
import numpy as np # type: ignore
from .reiform_exceptions import *

import cupy

def get_image_channels(path : str) -> int:

    im : Image.Image = Image.open(path)

    return len(im.getbands())

def get_image_metadata(path : str, convert_to_rgb : bool = False) -> dict:

    try:

        im : Image.Image = Image.open(path)

        width, height = im.size

        channels = len(im.getbands())

        if convert_to_rgb:
            im = im.convert("RGB")
        im_arr = np.array(im)

        if len(im_arr.shape) > 2:
            mean = list(np.mean(im_arr, (0, 1)))
            std = list(np.std(im_arr, (0, 1)))
        else:
            mean = list([np.mean(im_arr)])
            std = list([np.std(im_arr)])

        if im_arr.dtype == "uint8":
            mean = [m/255.0 for m in mean]
            std = [s/255.0 for s in std]

        return {
            "width": width,
            "height": height,
            "channels": channels,
            "mean": mean,
            "std_dev": std
        }
        
    except:
        raise ReiformFileSystemException("Couldn't get metadata from {}".format(path))

def closest_power_of_2(max_size):
    # Find the closest power of 2 for the edge size
    closest_size : int = 2
    while closest_size < max_size:
        closest_size *= 2
    return closest_size

def max_sizes_(data ):
    sizes : Tuple[int, int, int] = data.find_max_image_dims()
    max_size : int = max(sizes)
    return sizes, max_size
