from impl.services.modules.core.resources import *
from impl.services.modules.core.reiform_imageclassificationdataset import ReiformICDataSet
from python.impl.services.modules.core.reiform_imageclassificationdataset import ReiformICFile
from gap_detection import *
import uuid

def load_image(filename):
    # Load the image as an RGB image
    image = Image.open(filename).convert("RGB")
    # Return the image as a NumPy array
    return np.array(image)

def save_image(image, filename):
    # Convert the image to a PIL image
    image = Image.fromarray(image)
    # Save the image
    image.save(filename)