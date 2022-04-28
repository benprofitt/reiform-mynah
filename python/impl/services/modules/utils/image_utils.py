from typing import Tuple
from PIL import Image # type: ignore
import numpy as np
from .reiform_exceptions import *

def get_image_metadata(path : str) -> dict:

    try:

        im : Image.Image = Image.open(path)

        width, height = im.size

        channels = len(im.getbands())

        im_arr = np.array(im)

        if len(im_arr.shape) > 2:
            mean = list(np.mean(im_arr, (0, 1)))
            std = list(np.std(im_arr, (0, 1)))
        else:
            mean = list(np.mean(im_arr))
            std = list(np.std(im_arr))

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
    closest_size : int = 2
    while closest_size < max_size:
        closest_size *= 2
    return closest_size

def max_sizes(data):
    # Find the closest power of 2 for the edge size
    sizes : Tuple[int, int, int] = data.find_max_image_size()
    max_size : int = max(sizes)
    return sizes, max_size
