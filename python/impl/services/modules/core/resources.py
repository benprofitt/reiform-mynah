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
import warnings

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').disabled = True

random.seed(73434323)

image_extns : List[str] = ["png", "jpeg", "jpg", "tif", "tiff"]
WORKERS : int = 3
VERBOSE = False

PROJECTION_LABEL : str = "inception_projection"

EMBEDDING_MODEL_NAME : str = "densenet201-imagenet-torch.pt"

LOCAL_EMBEDDING_PATH_MOBILENET : str = "models/mobilenet-v2-embeddings.pt"
LOCAL_EMBEDDING_PATH_RESNET18 : str = "models/resnet18-imagenet-torch.pt"
LOCAL_EMBEDDING_PATH_RESNET50 : str = "models/resnet50-imagenet-torch.pt"
LOCAL_EMBEDDING_PATH_RESNET152 : str = "models/resnet152-imagenet-torch.pt"
LOCAL_EMBEDDING_PATH_INCEPTIONV3 : str = "models/inception-v3-imagenet-torch.pt"
LOCAL_EMBEDDING_PATH_DENSENET201 : str = "models/densenet201-imagenet-torch.pt"

LOCAL_PRETRAINED_PATH_LIGHT_DETECTION : str = "models/lighting/detection"
LOCAL_PRETRAINED_PATH_LIGHT_CORRECTION : str = "models/lighting/correction"

PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION = "PROJECTION_LABEL_FULL_EMBEDDING_CONCATENATION "
PROJECTION_LABEL_REDUCED_EMBEDDING = "PROJECTION_LABEL_REDUCED_EMBEDDING"
PROJECTION_LABEL_REDUCED_EMBEDDING_PER_CLASS = "PROJECTION_LABEL_REDUCED_EMBEDDING_PER_CLASS"
PROJECTION_LABEL_2D_PER_CLASS = "PROJECTION_LABEL_3D_PER_CLASS"
PROJECTION_LABEL_2D : str = "2D_projection"

LIGHTING_PREDICTION = "lighting_prediciton"
NEW_LABEL_PREDICTION = "corrected_label_prediction"
NEW_LABEL_PREDICTION_PROBABILITIES = "corrected_label_prediction_probability_vector"

MODEL_METADATA = "model_metadata"
PATH_TO_MODEL = "path_to_model"
MODEL = "model"
CHANNELS = "channels"
SIZE = "size"
CROP = "crop"
RESIZE = "resize"
MEAN = "mean"
STD = "std"
LATENT_SIZE = "latent_size"
NAME = "dataset_name"
IMAGE_DIMS = "image_dims"

EMBEDDING_DIM_SIZE = 31

VARIATIONAL_BETA = 0.0000001
VAE_PROJECTION_TRAINING_EPOCHS = 200

CORRECTION_MODEL_BATCH_SIZE = 512
MAX_CORRECTION_MODEL_BATCH_SIZE = 1024

MAX_EMBEDDING_MODEL_BATCH_SIZE = 2048
RESNET_SIZE = 299

# From Mislabeled Correction - need to be more dynamic
insize = 3
MONTE_CARLO_TRAINING_EPOCHS = 50
MONTE_CARLO_SIMULATIONS = 75
DATASET_EVAL_EPOCHS = 15

device = ("cuda" if torch.cuda.is_available() else "cpu")
# device = ("cpu") # Use this if you get obfuscated CUDA errors

BASE_EMBEDDING_MODEL_BATCH_SIZE = int(12 if device == "cpu" else (torch.cuda.mem_get_info(0)[0] * 1.75) // (1024 ** 3))
BASE_RESNET_50_MODEL_BATCH_SIZE = int(12 if device == "cpu" else (torch.cuda.mem_get_info(0)[0] * 2.5) // (1024 ** 3))
AVAILABLE_THREADS = 3

def empty_mem_cache():
    if device == "cuda":
        empty_mem_cache()

def load_pt_model(model: nn.Module, path : str):
    if device == "cuda":
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model

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