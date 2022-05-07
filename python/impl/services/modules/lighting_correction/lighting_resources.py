from impl.services.modules.core.resources import *
from impl.services.modules.core.embeddings.latent_projection import *
from impl.services.modules.core.vae_auto_net import *
from impl.services.modules.core.reiform_imageclassificationdataset import *

from impl.services.modules.core.reiform_models import *

import pyod # type: ignore
import pyod.models as odm # type: ignore
from pyod.models.loda import LODA # type: ignore

from PIL import ImageEnhance # type: ignore

# Sizing options for pretraining 
# TODO: More formalization
LIGHTING_DETECTION_EDGE_SIZES = [64, 128, 256, 512]
LIGHTING_CORRECTION_EDGE_SIZES = [64, 256, 1024]