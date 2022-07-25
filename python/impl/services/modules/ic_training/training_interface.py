from impl.services.modules.core.resources import *
from python.impl.services.modules.core.reiform_imageclassificationdataset import ReiformICDataSet
from python.impl.services.modules.core.reiform_models import multiclass_model_loss
from python.impl.services.modules.utils.progress_logger import ProgressLogger

class TrainingSpecifications:

    def __init__(self, epochs : int, min_epochs : int, optimizer : Any, 
                 loss_epsilon : float, batch_size : int, train_test_split : float, 
                 transformation : Any, loss_function : Callable) -> None:
        
        self.epochs = epochs
        self.min_epochs = min_epochs
        self.loss_epsilon = loss_epsilon
        self.batch_size = batch_size

        self.train_test_split = train_test_split

        self.optimizer = optimizer
        self.transformation = transformation
        self.loss_function = loss_function

def get_names_from_dataloader(dataloader : Any) -> List[str]:

    names : List[str] = []

    for _, _, name in dataloader:
        names.append(name)

    return names

def calculate_loss(model : nn.Module, dataloader : Any, loss_function : Callable) -> List[float]:

    losses : List[float] = []

    for image_batch, labels, _ in dataloader:
    
        image_batch = image_batch.to(device)
         
        # model reconstruction
        prediction = model(image_batch)
        labels = labels.to(device)
        # reconstruction error
        loss = loss_function(prediction, labels)

        losses.append(loss.item())

    return losses

def train_ic_model(dataset : ReiformICDataSet, model : nn.Module, training_specs: TrainingSpecifications, logger : ProgressLogger) -> Tuple[nn.Module, List[float]]:

    train_ds, test_ds = dataset.split(training_specs.train_test_split)

    # There should be some conditionals involved with these. Batch size should be limited to GPU capacity.
    batch_size = training_specs.batch_size
    transformation = training_specs.transformation

    train_dl_pt = train_ds.get_balanced_dataloader(3, 256, batch_size, transformation)
    test_dl_pt = test_ds.get_dataloader(3, 256, batch_size, transformation, shuffle=False)

    return train_model(model, training_specs, train_dl_pt, test_dl_pt, logger)

def train_model(model : nn.Module, training_specs : TrainingSpecifications, train_dl_pt, test_dl_pt, logger : ProgressLogger):
    loss_function = (training_specs.loss_function if training_specs.loss_function is not None else multiclass_model_loss)
    epsilon = training_specs.loss_epsilon
    model.train()
    model.to(device)

    train_loss_avg : List[float] = []
    test_losses : List[List[float]] = []
    
    logger.write("INFO  Begin training")

    for epoch in range(1, training_specs.epochs+1):
      logger.write("INFO  Start epoch {}/{}".format(epoch, training_specs.epochs+1))
      train_loss_avg.append(0)

      num_batches = 0
      
      for image_batch, labels, _ in train_dl_pt:
        image_batch = image_batch.to(device)
         
        # model reconstruction
        prediction = model(image_batch)
        labels = labels.to(device)
        # reconstruction error
        loss = loss_function(prediction, labels)
        
        # backpropagation
        training_specs.optimizer.zero_grad()
        loss.backward()
        
        # one step of the optmizer (using the gradients from backpropagation)
        training_specs.optimizer.step()

        train_loss_avg[-1] += loss.item()
        num_batches += 1

    
      test_losses.append(calculate_loss(model, test_dl_pt, loss_function))
      train_loss_avg[-1] /= num_batches

      # Stop condition (helps with wild training times on huge datasets)
      if len(train_loss_avg) > training_specs.min_epochs:
        delta_2 = train_loss_avg[-2] - train_loss_avg[-1]
        delta_1 = train_loss_avg[-3] - train_loss_avg[-2]
        if delta_1 < epsilon and delta_2 < epsilon and delta_2 > 0 and delta_1 > 0:
            break

    return model, train_loss_avg, test_losses, get_names_from_dataloader(test_dl_pt)
