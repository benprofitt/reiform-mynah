from PIL import Image # type: ignore
from .reiform_exceptions import *

def get_image_metadata(path : str) -> dict:

    try:

        im : Image.Image = Image.open(path)

        width, height = im.size

        channels = len(im.getbands())

        return {
            "width": width,
            "height": height,
            "channels": channels
        }
    
    except:
        raise ReiformFileSystemException("Couldn't get metadata from {}".format(path))