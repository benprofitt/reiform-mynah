import os, sys # type: ignore
import time # type: ignore
import random # type: ignore
import copy # type: ignore

from multiprocessing import Pool

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
from PIL import Image # type: ignore

import sklearn # type: ignore
from sklearn.ensemble import IsolationForest # type: ignore
from sklearn.decomposition import PCA  # type: ignore

from glob import glob # type: ignore

import torch # type: ignore
import torchvision # type: ignore
import torchvision.transforms as transforms # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

import typing
from typing import List, Tuple, Dict, Any, Callable, Optional
from nptyping import NDArray # type: ignore

from impl.services.modules.utils.reiform_exceptions import *

random.seed(7343432676)

image_extns : List[str] = ["png", "jpeg", "jpg", "tif", "tiff"]
workers : int = 4
VERBOSE = False

PROJECTION_LABEL : str = "inception_projection"
PROJECTION_LABEL_2D : str = "2D_projection"

LOCAL_EMBEDDING_PATH : str = "models/embedding"

PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION = "PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION "
PROJECTION_LABEL_REDUCED_EMBEDDING = "PROJECTION_LABEL_REDUCED_EMBEDDING"
PROJECTION_LABEL_REDUCED_EMBEDDING_PER_CLASS = "PROJECTION_LABEL_REDUCED_EMBEDDING_PER_CLASS"
PROJECTION_LABEL_3D_PER_CLASS = "PROJECTION_LABEL_3D_PER_CLASS"

CHANNELS = "channels"
SIZE = "size"
CROP = "crop"
RESIZE = "resize"
MEAN = "mean"
STD = "std"
LATENT_SIZE = "latent_size"
NAME = "dataset_name"

VARIATIONAL_BETA = 0.0000001
VAE_PROJECTION_TRAINING_EPOCHS = 100
CORRECTION_MODEL_BATCH_SIZE = 364

# From Mislabeled Correction - need to be more dynamic
insize = 3
edgesizes = [16, 32, 64]
monte_carlo_simulations = 85
monte_carlo_simulations = 5
epochs = 14
epochs = 5

device = ("cuda" if torch.cuda.is_available() else "cpu")
# device = ("cpu") # Use this if you get obfuscated CUDA errors

def get_folder_contents(path: str) -> List[str]:

    filenames : List[str] = []
    for root, subdirs, files in os.walk(path):
        for name in files:
            if name.split(".")[-1].lower() in image_extns:
                filenames.append("{}/{}".format(root, name))

    return filenames

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#