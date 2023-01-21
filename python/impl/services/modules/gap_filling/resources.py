from matplotlib.path import Path
from impl.services.modules.core.resources import *
from impl.services.modules.core.reiform_imageclassificationdataset import ReiformICDataSet
from impl.services.modules.core.reiform_imageclassificationdataset import ReiformICFile
from impl.services.modules.utils.reiform_exceptions import ReiformInfo
from gap_detection import *
import uuid, shutil
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector

from impl.services.modules.gap_filling.image_smoothing import ImageSmoother

def plot_scatter(points_list):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] # define a list of colors
    for i, points in enumerate(points_list):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.scatter(x, y, c=colors[i % len(colors)]) # use the color from the list, and wrap around if there are more points than colors
    plt.show()


GAP_PROJECTION_LABEL = PROJECTION_LABEL_2D_PER_CLASS

def load_image(filename):
    # Load the image as an RGB image
    image = Image.open(filename).convert("RGB")
    # Return the image as a NumPy array
    return np.array(image)

def save_image(image, filename):
    # Convert the image to a PIL image
    smoother = ImageSmoother()
    image = smoother.forward(image)

    # Save the image
    image.save(filename)

def save_images_from_files(files : List[ReiformICFile], cluster_num : int):
    
    if not len(files):
        return

    base_path = "cluster_images/{}/{}/".format(files[0].get_class(), str(cluster_num))

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for file in files:
        shutil.copy2(file.get_name(), base_path)


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

def plot_scatter_(*points_list):
    n = len(points_list)
    fig, axes = plt.subplots(1, n) # create a figure with n subplots
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] # define a list of colors
    if n == 1:
        axes = [axes]
    for ax, points in zip(axes, points_list):
        for i, point in enumerate(points):
            x = [p[0] for p in point]
            y = [p[1] for p in point]
            ax.scatter(x, y, c=colors[i % len(colors)]) # use the color from the list, and wrap around if there are more points than colors
    plt.show()


def plot_embeddings_multi(datasets : List[ReiformICDataSet], label : str, classes : List[str]):
    '''TODO: Make this work with datasets that don't have classes that are numbers.'''
    '''This is really just for testing and research!'''

    n = len(datasets)
    fig, axes = plt.subplots(1, n) # create a figure with n subplots
    if n == 1:
        axes = [axes]


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

    color_offset = 0

    for ax, dataset in zip(axes, datasets):
        X : List[float] = []
        Y : List[float] = []
        c : List[str] = []
        for file in dataset.all_files():

            if file.get_class() in classes:

                if "new_images/" in file.get_name():
                    c.append(str((int(file.get_class()) + color_offset * 2) % 20))
                else:
                    c.append(str(int(file.get_class()) + color_offset))
                proj = file.get_projection(label)
                X.append(proj[0])
                Y.append(proj[1])

        c = [color_map[v] for v in c]
        ax.scatter(X, Y, c=c)

        color_offset += len(dataset.classes())

    plt.show()



def generate_filename():
    uuid4 = uuid.uuid4()
    new_filename = "new_images/{}.png".format(uuid4)

    return new_filename

def combine_image_averages(file1, file2):
    # load the two files
    im1 = load_image(file1.get_name())
    im2 = load_image(file2.get_name())

    # join them in some way? to make new image
    im_new = (im1//2 + im2//2)
    
    # Generate a random UUID for the name - uuid4 gives more privacy that uuid1
    new_filename = generate_filename()
    save_image(im_new, new_filename)
    return new_filename

def combine_images_patches(image_list : List[ReiformICFile]):
    image_list = [im.get_name() for im in image_list]
    # Open the images using PIL
    img1 = Image.open(image_list[0])
    img_list = [Image.open(img) for img in image_list[1:]]
    # Get the width and height of the images
    width, height = img1.size

    for img in img_list:
        # Select random patches of the images
        x1, y1 = random.randint(0, width-1), random.randint(0, height-1)
        x2, y2 = random.randint(x1, width-1), random.randint(y1, height-1)
        img_patch = img.crop((x1, y1, x2, y2))

        # paste the patch onto img1 in the same location
        img1.paste(img_patch, (x1, y1))

    # Save the new image
    filename = generate_filename()
    save_image(img1, filename)

    return filename

def new_images_grey_patch(image_list : List[ReiformICFile]):

    filenames = []

    for im in image_list:
        image = im.get_name()

        # Open the image using PIL
        img = Image.open(image)

        # Get the width and height of the image
        width, height = img.size

        # Calculate the area of the image
        area = width * height

        # Select a random patch between 1/6 and 1/4 of the entire image in area
        patch_area = random.randint(int(area/6), int(area/4))
        patch_width = int(patch_area / height)
        patch_height = int(patch_area / width)
        x1, y1 = random.randint(0, width-patch_width), random.randint(0, height-patch_height)
        x2, y2 = x1 + patch_width, y1 + patch_height
        patch = img.crop((x1, y1, x2, y2))

        # Create a new image filled with the grey color
        grey = (128, 128, 128)
        new_img = Image.new("RGB", patch.size, grey)

        # Paste the new image in the same location as the patch
        img.paste(new_img, (x1, y1))

        # Save the new image
        new_filename = generate_filename()

        save_image(img, new_filename)

        filenames.append(new_filename)

    return filenames

def average_image(image_list : List[ReiformICFile]):
    # Files to names (to be opened)
    image_list = [im.get_name() for im in image_list]

    # Open ims and divide by len (for taking avg)
    images = [load_image(im)//len(image_list) for im in image_list]

    # Add images
    new_image = images[0]
    for im in images[1:]:
        new_image += im

    new_image = Image.fromarray(new_image)

    fname = generate_filename()
    save_image(new_image, fname)

    return [fname]

def swap_pixel(image_list : List[ReiformICFile]):

    # Images to names
    image_list = [im.get_name() for im in image_list]

    # Open the images using PIL
    imgs = [Image.open(img) for img in image_list]

    # Get the width and height of the images
    widths = [img.size[0] for img in imgs]
    heights = [img.size[1] for img in imgs]

    for i in range(len(imgs)):
        # Select a random pixel from the current image
        x1, y1 = random.randint(0, widths[i]-1), random.randint(0, heights[i]-1)
        # Get the pixel color from the current image
        pixel1 = imgs[i].getpixel((x1, y1))
        # Get the next image in the rotation
        next_img = imgs[(i+1)%len(imgs)]
        # Select a random pixel from the next image
        x2, y2 = random.randint(0, next_img.size[0]-1), random.randint(0, next_img.size[1]-1)
        # Get the pixel color from the next image
        pixel2 = next_img.getpixel((x2, y2))
        # Put the pixel color of the current image to the next image
        next_img.putpixel((x2, y2), pixel1)
        # Put the pixel color of the next image to the current image
        imgs[i].putpixel((x1, y1), pixel2)

    filenames = []
    # Save the new images
    for i,img in enumerate(imgs):
        name = generate_filename()

        save_image(img, name)
        
        filenames.append(name)
    
    return filenames

def get_embedding_from_file(pack : Tuple[str, ReiformICFile]):
    file = pack[1]
    return file.get_projection(GAP_PROJECTION_LABEL)

def select_region(objects, object_to_point_fn):
    # Use the conversion function to get a list of points
    points = [object_to_point_fn(obj) for obj in objects]

    # Plot the points on a scatter plot
    plt.scatter(*zip(*points))

    # Create lists to store the selected and unselected objects
    selected_objects = []
    unselected_objects = []

    # Define a callback function for the lasso selector
    def onselect(verts):
        # Get the indices of the selected points
        # selected_indices = [i for i, point in enumerate(points) if np.isclose(point, verts, atol=1e-5).any(1).any()]
        path = Path(verts)
        # Get the indices of the points inside the lasso
        selected_indices = [i for i, point in enumerate(points) if path.contains_point(point)]

        # Append the corresponding objects to the selected list
        selected_objects.extend([objects[i] for i in selected_indices])
        unselected_objects.extend([obj for obj in objects if obj not in selected_objects])
    # Create the lasso selector
    selector = LassoSelector(plt.gca(), onselect)
    plt.show()
    return selected_objects, unselected_objects


def make_gapped_dataset(train_dataset : ReiformICDataSet, test_dataset : ReiformICDataSet):

    new_train_ds : ReiformICDataSet = ReiformICDataSet(train_dataset.class_list)

    for cls in train_dataset.class_list:

        files = []
        for k, f in train_dataset.get_items(cls):
            files.append((k, f))
        
        in_, out_ = select_region(files, get_embedding_from_file)
        for _, f in out_:
            new_train_ds.add_file(f)
        for _, f in in_:
            test_dataset.add_file(f)

    return new_train_ds, test_dataset