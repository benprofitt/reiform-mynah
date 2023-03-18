from resources import *
from enum import Enum
from functools import partial

from art.attacks.evasion import FastGradientMethod, ShadowAttack, PixelAttack, SaliencyMapMethod
from art.estimators.classification import PyTorchClassifier

SAVE_PATH = "LOCAL_IMAGE_STORAGE"
BATCH_SIZE_ATTACK_GENERATION = 96

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

    classifier = create_art_classifier()

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
        filename = "{}/{}".format(SAVE_PATH, uuid_name)

        Image.fromarray(np.uint8(adv_image*255)).save(filename)

        reiform_file = ReiformICFile(uuid_name, label)
        reiform_file.uuid = new_uuid
        new_dataset.add_file(reiform_file)

        generated_image_data.append((uuid_name, new_uuid, name, label))

    return new_dataset

def create_art_classifier():
    model_path = LOCAL_EMBEDDING_PATH_MOBILENET

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
    
    return classifier

def get_art_dataloader(dataset : ReiformICDataSet):
    return dataset.get_dataloader(in_size=3, edge_size=224, batch_size=BATCH_SIZE_ATTACK_GENERATION, shuffle=False)

def pixel_attack(dataset : ReiformICDataSet) -> ReiformICDataSet:

    # Each tuple holds: new uuid, new filename, old filename, original class
    generated_image_data : List[Tuple[str, str, str, str]] = []
    new_dataset = ReiformICDataSet(dataset.class_list)

    classifier = create_art_classifier()
    attack = PixelAttack(classifier)

    pt_dataloader = get_art_dataloader(dataset)

    for images, labels, names in pt_dataloader:

        adv_images = attack.generate(images.numpy())

        for image, label, name in zip(adv_images, labels, names):
            adv_image = np.transpose(image[0], (1, 2, 0))

            uuid_name, new_uuid = generate_filename_pair()
            filename = "{}/{}".format(SAVE_PATH, uuid_name)

            Image.fromarray(np.uint8(adv_image*255)).save(filename)

            reiform_file = ReiformICFile(uuid_name, label)
            reiform_file.uuid = new_uuid
            new_dataset.add_file(reiform_file)

            generated_image_data.append((uuid_name, new_uuid, name, label))
    
    return new_dataset

def patch_attack(dataset : ReiformICDataSet) -> ReiformICDataSet:

    ReiformUnimplementedException("Patch attack is too complicated to implement for a PoC")

def JSM_attack(dataset : ReiformICDataSet) -> ReiformICDataSet:

    # Each tuple holds: new uuid, new filename, old filename, original class
    generated_image_data : List[Tuple[str, str, str, str]] = []
    new_dataset = ReiformICDataSet(dataset.class_list)

    classifier = create_art_classifier()
    attack = SaliencyMapMethod(classifier, BATCH_SIZE_ATTACK_GENERATION)

    pt_dataloader = get_art_dataloader(dataset)

    for images, labels, names in pt_dataloader:

        adv_images = attack.generate(images.numpy())

        for image, label, name in zip(adv_images, labels, names):
            adv_image = np.transpose(image[0], (1, 2, 0))

            uuid_name, new_uuid = generate_filename_pair()
            filename = "{}/{}".format(SAVE_PATH, uuid_name)

            Image.fromarray(np.uint8(adv_image*255)).save(filename)

            reiform_file = ReiformICFile(uuid_name, label)
            reiform_file.uuid = new_uuid
            new_dataset.add_file(reiform_file)

            generated_image_data.append((uuid_name, new_uuid, name, label))
    
    return new_dataset

def get_candidates(dataset : ReiformICDataSet) -> ReiformICDataSet:

    # Select images that are in prime locations for evaluation to have adversarial pairs created and evaluated
    new_dataset = ReiformICDataSet(dataset.classes())
    
    # Let's start with random selection for now and do better later
    p = 0.1

    for c in dataset.classes():
        for k, f in dataset.get_items(c):
            if random.random() < p:
                new_dataset.add_file(f)

    return new_dataset

def generate_pairs(dataset : ReiformICDataSet, 
                   methods : List[AdversarialGenerationMethod]) -> Dict[AdversarialGenerationMethod,
                                                                                  ReiformICDataSet]:
    
    resulting_datasets : Dict[AdversarialGenerationMethod, ReiformICDataSet] = {}

    for method in methods:
        resulting_datasets[method] = method(dataset)

    return resulting_datasets

