from resources import *
from enum import Enum
from functools import partial

from art.attacks.evasion import FastGradientMethod, ShadowAttack
from art.estimators.classification import PyTorchClassifier

class AdversarialGenerationMethod(Enum):

    pixel_attack = partial(pixel_attack)
    shadow_attack = partial(shadow_attack)
    patch_attack = partial(patch_attack)
    JSM_attack = partial(JSM_attack)

    def __call__(self, *args):
        return self.value(*args)

def create_shadow_image(attack: ShadowAttack, x_test: NDArray, y_test: NDArray):
    x_test_adv = attack.generate(x=x_test, y=y_test)
    return x_test_adv

def shadow_attack(dataset : ReiformICDataSet) -> ReiformICDataSet:

    # Each tuple holds: new uuid, new filename, old filename, original class
    generated_image_data : List[Tuple[str, str, str, str]] = []
    new_dataset = ReiformICDataSet(dataset.class_list)

    model_path = LOCAL_EMBEDDING_PATH_MOBILENET
    save_path = "LOCAL_IMAGE_STORAGE"

    model, optimizer, criterion = load_checkpoint_to_attack(model_path)

    # Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=1000,
    )

    attack = ShadowAttack(estimator=classifier, batch_size=96, nb_steps=100)

    pt_dataset = dataset.get_dataset(in_size=3, edge_size=224)

    ds_len = len(pt_dataset)

    for i in range(0, ds_len):

        im, label, name = pt_dataset[i]
        single_image = im.unsqueeze(0).numpy()
        predictions_true = classifier.predict(single_image)
        y_value = np.argmax(predictions_true[0])

        adv_image = create_shadow_image(attack, single_image, y_value)
        adv_image = np.transpose(adv_image[0], (1, 2, 0))


        uuid_name, new_uuid = generate_filename_pair()
        filename = "{}/{}".format(save_path, uuid_name)

        Image.fromarray(np.uint8(adv_image*255)).save(filename)

        reiform_file = ReiformICFile(uuid_name, label)
        reiform_file.uuid = new_uuid
        new_dataset.add_file(reiform_file)

        generated_image_data.append((uuid_name, new_uuid, name, label))

    return new_dataset


def pixel_attack(dataset : ReiformICDataSet) -> ReiformICDataSet:
    pass

def patch_attack(dataset : ReiformICDataSet) -> ReiformICDataSet:
    pass

def JSM_attack(dataset : ReiformICDataSet) -> ReiformICDataSet:
    pass


def get_candidates(dataset : ReiformICDataSet) -> ReiformICDataSet:

    # Select images that are in prime locations for evaluation to have adversarial pairs created and evaluated

    pass


def generate_pairs(dataset : ReiformICDataSet, 
                   methods : Dict[AdversarialGenerationMethod, Callable]) -> Dict[AdversarialGenerationMethod,
                                                                                  ReiformICDataSet]:
    
    resulting_datasets : Dict[AdversarialGenerationMethod, ReiformICDataSet] = {}

    for method in methods:
        resulting_datasets[method] = method(dataset)

    return resulting_datasets

