from impl.services.modules.core.resources import *
from impl.services.modules.core.reiform_imageclassificationdataset import ReiformICDataSet
from impl.services.modules.utils.model_utils import calculate_batch_size

def process_dataset(dataset : ReiformICDataSet, model : nn.Module, transform : Any, dims : List[int]) -> ReiformICDataSet:
    
    model.to(device)
    model.eval()

    batch_size = calculate_batch_size(dims)
    pt_dataloader = dataset.get_dataloader_with_names(transform, batch_size)


    for batch, labels, names in pt_dataloader:

        batch = batch.to(device)
         
        # model reconstruction
        predictions = model(batch)

        for i, name in enumerate(names):

            pred_vec = predictions[i]
            pred = torch.argmax(pred_vec)

            old_label = dataset.classes()[labels[i]]

            dataset.set_file_class(old_label, name, dataset.classes()[int(pred)])
            dataset.get_file(old_label, name).add_confidence_vector(pred_vec.to("cpu").detach().numpy())

    return dataset