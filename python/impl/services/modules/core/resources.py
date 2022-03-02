import os, sys # type: ignore
import time # type: ignore
import random # type: ignore
import copy # type: ignore

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
from PIL import Image # type: ignore

import sklearn # type: ignore
from sklearn.ensemble import IsolationForest # type: ignore

import torch # type: ignore
import torchvision # type: ignore
import torchvision.transforms as transforms # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

import typing
from typing import List, Tuple, Dict, Any, Callable
from nptyping import NDArray # type: ignore

from impl.services.modules.utils.reiform_exceptions import *

random.seed(7343676)

image_extns : List[str] = ["png", "jpeg", "jpg", "tif", "tiff"]
workers : int = 4
VERBOSE = False

PROJECTION_LABEL : str = "inception_projection"
PROJECTION_LABEL_2D : str = "2D_projection"

device = ("cuda" if torch.cuda.is_available() else "cpu")
# device = ("cpu") # I use this what I get obfuscated CUDA errors

def get_folder_contents(path: str) -> List[str]:

    filenames : List[str] = []
    for root, subdirs, files in os.walk(path):
        for name in files:
            if name.split(".")[-1].lower() in image_extns:
                filenames.append("{}/{}".format(root, name))

    return filenames


#