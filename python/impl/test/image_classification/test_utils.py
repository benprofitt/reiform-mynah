from sklearn.metrics import precision_recall_fscore_support # type: ignore
from impl.services.modules.utils.image_utils import closest_power_of_2, max_sizes
from impl.services.modules.core.reiform_imageclassificationdataset import *
from impl.services.modules.core.reiform_models import AutoNet, train_conv_net


def get_predictions(dataloader, model):

    model = model.to(device)
    model.eval()

    labels = []
    predictions = []

    for image, label in dataloader:
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

def dataset_evaluation(train_ds : ReiformICDataSet, test_ds : ReiformICDataSet):

    sizes, max_ = max_sizes(train_ds)
    edge_size = closest_power_of_2(max_)
    batch_size = 256
    epochs = 25
    classes = len(train_ds.classes())

    transformation = transforms.Compose([
        transforms.Resize((edge_size, edge_size)),
        transforms.RandomCrop(edge_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_ds.get_mean(), std=test_ds.get_std_dev())
    ])

    train_dl_pt = train_ds.get_dataloader(sizes[2], edge_size, batch_size, transformation)
    test_dl_pt = test_ds.get_dataloader(sizes[2], edge_size, batch_size, transformation)

    return train_model_for_eval(train_dl_pt, test_dl_pt, sizes, edge_size, epochs, classes)

def train_model_for_eval(train_dl_pt, test_dl_pt, sizes, edge_size, epochs, classes):
    
    learning_rate = 0.001
    momentum = 0.95

    model = AutoNet(sizes[2], edge_size, classes)

    loss = F.cross_entropy
    optim = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-2)

    model, _ = train_conv_net(model, train_dl_pt, loss, optim, epochs)

    true_y, pred_y = get_predictions(test_dl_pt, model)

    scores = precision_recall_fscore_support(true_y, pred_y, average='macro')

    return scores
