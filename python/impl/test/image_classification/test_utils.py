from sklearn.metrics import precision_recall_fscore_support # type: ignore
from impl.services.modules.utils.image_utils import closest_power_of_2
from impl.services.modules.core.reiform_imageclassificationdataset import *
from impl.services.modules.core.reiform_models import AutoResnet, DeepAutoNet, train_conv_net


def get_predictions(dataloader, model):

    model = model.to(device)
    model.eval()

    labels = []
    predictions = []

    for image, label, _ in dataloader:
        image = image.to(device)
        pred = model(image)
        for i in range(len(label)):    
            val = int(torch.argmax(pred[i]))
            lab = int(label[i])
            predictions.append(val)
            labels.append(lab)

    return labels, predictions

def get_predictions_with_names(dataloader, model):

    model = model.to(device)
    model.eval()

    labels = []
    predictions = []
    names = []

    for image, label, name in dataloader:
        image = image.to(device)
        pred = model(image)
        for i in range(len(label)):    
            val = int(torch.argmax(pred[i]))
            lab = int(label[i])
            predictions.append(val)
            labels.append(lab)
            names.append(name[i])

    return labels, predictions, names

def dataset_evaluation_resnet(train_ds : ReiformICDataSet, test_ds : ReiformICDataSet):

    batch_size = BASE_RESNET_50_MODEL_BATCH_SIZE
    epochs = DATASET_EVAL_EPOCHS
    classes = len(train_ds.classes())

    transformation = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_ds.get_mean(), std=train_ds.get_std_dev())
    ])

    train_dl_pt = train_ds.get_balanced_dataloader(3, 256, batch_size, transformation)
    test_dl_pt = test_ds.get_dataloader(3, 256, batch_size, transformation)

    return train_model_for_evaluation_resnet(train_dl_pt, test_dl_pt, epochs, classes)

def train_model_for_evaluation_resnet(train_dl_pt, test_dl_pt, epochs : int, classes : int):

    learning_rate = 0.001

    model = AutoResnet(classes)

    loss = F.cross_entropy
    
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-2)

    model, _ = train_conv_net(model, train_dl_pt, loss, optim, epochs, epsilon=0.00001)

    scores = get_scores_for_model(test_dl_pt, model)

    return scores

def get_scores_for_model(test_dl_pt, model):
    true_y, pred_y = get_predictions(test_dl_pt, model)

    scores = precision_recall_fscore_support(true_y, pred_y, average=None)
    return scores

def dataset_evaluation(train_ds : ReiformICDataSet, test_ds : ReiformICDataSet):

    sizes = train_ds.find_max_image_dims()
    max_ = max(sizes)
    edge_size = min(1024, closest_power_of_2(max_)*2)
    batch_size = 512
    epochs = DATASET_EVAL_EPOCHS
    classes = len(train_ds.classes())

    transformation = transforms.Compose([
        transforms.Resize((edge_size, edge_size)),
        transforms.RandomCrop(edge_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_ds.get_mean(), std=train_ds.get_std_dev())
    ])

    train_dl_pt = train_ds.get_dataloader(sizes[2], edge_size, batch_size, transformation)
    test_dl_pt = test_ds.get_dataloader(sizes[2], edge_size, batch_size, transformation)

    return train_model_for_eval(train_dl_pt, test_dl_pt, sizes, edge_size, epochs, classes)

def train_model_for_eval(train_dl_pt, test_dl_pt, sizes, edge_size, epochs, classes):
    
    learning_rate = 0.001
    momentum = 0.95

    model = DeepAutoNet(sizes[2], edge_size, classes)

    loss = F.cross_entropy
    
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-2)

    model, _ = train_conv_net(model, train_dl_pt, loss, optim, epochs, epsilon=0.00001)

    true_y, pred_y = get_predictions(test_dl_pt, model)

    scores = precision_recall_fscore_support(true_y, pred_y, average=None)

    return scores

