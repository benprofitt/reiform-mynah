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

