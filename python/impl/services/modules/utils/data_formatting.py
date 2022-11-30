import json
import os
from glob import glob
from impl.services.modules.core.resources import *
from impl.services.modules.core.reiform_imageclassificationdataset import ReiformICDataSet, ReiformICFile, make_file, make_file_with_RGB


def read_row(line):
    image_id, _, label, _, xmin, xmax, ymin, ymax, occl, trnc, grup, dpct, insd = line.split(",")

    return "{}.jpg".format(image_id), label, \
            (float(xmax)-float(xmin))*(float(ymax)-float(ymin)), \
            int(occl), int(trnc), int(grup), int(dpct), int(insd)

def h(f1, f2):
    if f1[2] > f2[2]:
        return f1
    return f2

def get_names(path):
    result = {}
    for x in os.walk(path):
        for y in glob(os.path.join(x[0], '*.jpg')):
            result[y.split("/")[-1]] = y

    return result

def read_csv_file(path, files_path, save_path):

    images : Dict[str, (str, str, float, bool, bool, bool, bool, bool)] = {}
    labels = {}
    names = get_names(files_path)

    start = time.time()

    with open(path, "r") as fh:
        
        for _ in fh:
            break
        for line in fh:
            f1 = read_row(line)
            labels[f1[1]] = 1
            if f1[0] not in images:
                images[f1[0]] = f1
            else:
                images[f1[0]] = h(images[f1[0]], f1)


    dataset = ReiformICDataSet(list(labels.keys()))
    packages = []
    for name, info in images.items():
        packages.append((names[name], info[1]))
    
    with Pool(AVAILABLE_THREADS) as p:
        files = p.map(make_file_with_RGB, packages)

    for file in files:
        dataset.add_file(file)

    dataset.find_max_image_dims(recalc=True)
    dataset.get_mean()

    ReiformInfo("Time to load dataset: {}".format(time.time() - start))

    with open(save_path, "w") as fh:
        json.dump(dataset.to_json(), fh, indent=2)

    start = time.time()

    bd = {}
    with open(save_path, 'r') as fh:
        bd = json.load(fh)

    new_dataset = ReiformICDataSet(bd["class_list"])
    new_dataset.from_json(bd)

    new_dataset.to_json()

    ReiformInfo("Time to deserialize dataset: {}".format(time.time() - start))
    
def load_dataset(path):
    with open(path, 'r') as fh:
        bd = json.load(fh)

    new_dataset = ReiformICDataSet(bd["class_list"])
    new_dataset.from_json(bd)

    return new_dataset
    
def save_dataset(path, ds):
    with open(path, "w") as fh:
        json.dump(ds.to_json(), fh, indent=2)

def flatten(name, ld):
    res = {}
    other_res = []

    subs = "Subcategory"
    l = "LabelName"

    for d in ld:
        res[d[l]] = d[l]
        if subs in d:
            osr = flatten(d[l], d[subs])
            sr = osr[0][1]
            for k in sr:
                res[k] = k
            other_res += osr

    return [(name, res)] + other_res
    

def create_dataset_subsets_via_hierarchy(dataset : ReiformICDataSet, 
                                         json_body : dict, save_path : str):

    flattened_structures = flatten(json_body["LabelName"], json_body["Subcategory"])

    for item in flattened_structures[1:]:
        label, category = item

        nds = ReiformICDataSet(list(category.keys()))
        for c in category:
            if c in dataset.classes():
                for _, file in dataset.get_items(c):
                    nds.add_file(file)
        nds.get_mean()
        nds.find_max_image_dims(True)
        full_path = "{}/{}.reiform_ds".format(save_path, label.replace("/m/", "openIm_"))
        with open(full_path, "w") as fh:
            json.dump(nds.to_json(), fh, indent=2)

def convert_to_3_main():
    dataset_path = sys.argv[1]

    ds = load_dataset(dataset_path)
    ds.max_channels = 3
    save_dataset(dataset_path, ds)

def subset_main():
    dataset_path = sys.argv[1]
    json_path = sys.argv[2]
    save_path = sys.argv[3]

    bd = {}
    with open(json_path, 'r') as fh:
        bd = json.load(fh)

    dataset = load_dataset(dataset_path)

    create_dataset_subsets_via_hierarchy(dataset, bd, save_path)

def main():
    csv_path = sys.argv[1]
    files = sys.argv[2]
    save = sys.argv[3]

    read_csv_file(csv_path, files, save)

if __name__ == "__main__":
    convert_to_3_main()