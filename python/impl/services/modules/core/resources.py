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
from sklearn.covariance import EllipticEnvelope # type: ignore
from sklearn.neighbors import LocalOutlierFactor # type: ignore

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

import umap # type: ignore

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

random.seed(73434323)

image_extns : List[str] = ["png", "jpeg", "jpg", "tif", "tiff"]
WORKERS : int = 3
VERBOSE = False

PROJECTION_LABEL : str = "inception_projection"
PROJECTION_LABEL_2D : str = "2D_projection"

LOCAL_EMBEDDING_PATH : str = "models/embedding"
LOCAL_EMBEDDING_PATH_MOBILENET : str = "models/mobilenet-v2-embeddings.pt"
LOCAL_PRETRAINED_PATH_LIGHT_DETECTION : str = "models/lighting/detection"
LOCAL_PRETRAINED_PATH_LIGHT_CORRECTION : str = "models/lighting/correction"

PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION = "PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION "
PROJECTION_LABEL_REDUCED_EMBEDDING = "PROJECTION_LABEL_REDUCED_EMBEDDING"
PROJECTION_LABEL_REDUCED_EMBEDDING_PER_CLASS = "PROJECTION_LABEL_REDUCED_EMBEDDING_PER_CLASS"
PROJECTION_LABEL_2D_PER_CLASS = "PROJECTION_LABEL_3D_PER_CLASS"

LIGHTING_PREDICTION = "lighting_prediciton"

CHANNELS = "channels"
SIZE = "size"
CROP = "crop"
RESIZE = "resize"
MEAN = "mean"
STD = "std"
LATENT_SIZE = "latent_size"
NAME = "dataset_name"

EMBEDDING_DIM_SIZE = 32

VARIATIONAL_BETA = 0.0000001
VAE_PROJECTION_TRAINING_EPOCHS = 200

CORRECTION_MODEL_BATCH_SIZE = 512
MAX_CORRECTION_MODEL_BATCH_SIZE = 1024

BASE_EMBEDDING_MODEL_BATCH_SIZE = 22
MAX_EMBEDDING_MODEL_BATCH_SIZE = 2048

# From Mislabeled Correction - need to be more dynamic
insize = 3
MONTE_CARLO_SIMULATIONS = 50
MONTE_CARLO_TRAINING_EPOCHS = 50

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