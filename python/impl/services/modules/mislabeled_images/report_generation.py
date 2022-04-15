from impl.services.modules.mislabeled_images.mislabeled_resources import *

def plot_in_2D(projections : Tuple[ReiformICDataSet], label : str) -> None:
    
    if len(projections[0].classes()) == 0:
        return

    x : Dict[int, List[float]] = {}
    y : Dict[int, List[float]] = {}
    c : Dict[int, List[str]] = {}

    classes = projections[0].classes()

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
    colors = [str(k) for k in range(20)]

    for i, clss in enumerate(classes):
        pass

    offset : int = 0
    for proj in projections:

        for class_name in proj.classes():
            data_points : ReiformICDataSet = proj.filter_classes(class_name)
            idx = classes.index(class_name)

            if idx not in x:
                x[idx] = []
                y[idx] = []
                c[idx] = []

            for file_name, vals in data_points.get_items(class_name):

                x[idx].append(vals.get_projection(label)[0])
                y[idx].append(vals.get_projection(label)[1])
                c[idx].append(color_map[str(idx + offset)])

        offset += 10


    fig = plt.figure()

    axes : List[Any] = []
    for idx in range(len(classes)):
        exec("ax{} = fig.add_subplot(2,5,idx + 1)".format(idx))
        exec("ax{}.scatter(x[idx], y[idx], c=c[idx])".format(idx))
        exec("axes.append(ax{})".format(idx))

    plt.show()

    # plt.rcParams['figure.figsize'] = [10, 10]
    # for i in range(len(classes)):
    #     plt.scatter(x[i], y[i], c=c[i])
    # plt.show()