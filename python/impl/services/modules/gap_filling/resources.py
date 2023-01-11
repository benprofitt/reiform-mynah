from impl.services.modules.core.resources import *
from impl.services.modules.core.reiform_imageclassificationdataset import ReiformICDataSet
from impl.services.modules.core.reiform_imageclassificationdataset import ReiformICFile
from gap_detection import *
import uuid

GAP_PROJECTION_LABEL = PROJECTION_LABEL_2D_PER_CLASS

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

def plot_embeddings(dataset : ReiformICDataSet, label : str, classes : List[str]):

    X : List[float] = []
    Y : List[float] = []
    c : List[str] = []

    for file in dataset.all_files():

        if file.get_class() in classes:

            c.append(file.get_class())
            proj = file.get_projection(label)
            X.append(proj[0])
            Y.append(proj[1])

    color_map : Dict[str, str] = {
        "0" : "red",
        "1" : "blue",
        "2" : "green",
        "3" : "yellow",
        "4" : "orange",
        "5" : "pink",
        "6" : "purple",
        "7" : "dimgray",
        "8" : "tan",
        "9" : "aqua",
        "10": "firebrick",
        "11": "royalblue",
        "12": "lime",
        "13": "gold",
        "14": "navajowhite",
        "15": "deeppink",
        "16": "mediumorchid",
        "17": "silver",
        "18": "peachpuff",
        "19": "darkcyan"
    }

    c = [color_map[v] for v in c]

    plt.scatter(X, Y, c=c)
    plt.show()