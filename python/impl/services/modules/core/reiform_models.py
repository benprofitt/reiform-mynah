from turtle import forward
from .model_resources import *

def create_conv_maxpool_block(insize: int, outsize: int):
    midsize: int = (insize + outsize)//2
    return nn.Sequential(
        create_conv_block(insize, midsize, 3, 1, 1),
        create_conv_block(midsize, midsize, 3, 1, 1),
        create_conv_block(midsize, outsize, 5, 1, 2),
        nn.MaxPool2d(2, stride=2)
    )

def create_small_conv_maxpool_block(insize: int, outsize: int):
    midsize: int = (insize + outsize)//2
    return nn.Sequential(
        create_conv_block(insize, midsize, 3, 1, 1),
        create_conv_block(midsize, outsize, 3, 1, 1),
        nn.MaxPool2d(2, stride=2)
    )

def create_conv_dilation_block(insize: int, outsize: int):
    midsize: int = (insize + outsize)//2
    return nn.Sequential(
        create_conv_block(insize, midsize, 1, 1, 0),
        create_conv_block(midsize, outsize, 3, 1, 2, d=2),
    )

def create_all_conv_blocks(insize: int, edgesize: int, linearsize: int, 
                           conv_block: Callable=create_conv_dilation_block, 
                           pool_block: Callable=create_conv_maxpool_block) -> nn.Sequential:

    max_size : int = 64**3

    curr_edge : int = edgesize
    curr_width : int = insize

    conv_blocks : List[nn.Sequential] = []

    next_width : int = max_size//(edgesize**2)

    while curr_edge * curr_edge * next_width > linearsize and curr_edge > 2:
        conv_blocks.append(
            conv_block(curr_width, next_width)
        )
        curr_width = next_width
        next_width //=2
        conv_blocks.append(
            pool_block(curr_width, next_width)
        )
        curr_width = next_width
        conv_blocks.append(
            conv_block(curr_width, curr_width)
        )
        conv_blocks.append(
            pool_block(curr_width, curr_width)
        )

        curr_edge //=4

    conv_blocks.append(nn.Sequential( nn.Flatten(), linear_block(curr_edge * curr_edge * curr_width,
                                    linearsize, dropout=0.1)))

    sequential = nn.Sequential(*conv_blocks)

    return sequential

def create_all_linear_blocks(insize : int, classes : int):

    curr_size : int = insize
    linear_blocks : List[nn.Sequential] = []
    while curr_size//10 > classes:
        linear_blocks.append(linear_block(curr_size, curr_size//3, dropout=0.1))
        curr_size = curr_size//3

    linear_blocks.append(linear_block(curr_size, classes, dropout=0.0, relu=False))
    return nn.Sequential(*linear_blocks)

class AutoNet(nn.Module):

    def __init__(self, insize: int, edgesize: int, classes: int) -> None:
        super().__init__()

        linear_size : int = min(4096, 400*classes)
        self.conv : nn.Sequential = create_all_conv_blocks(insize, edgesize, linear_size)

        self.fc : nn.Sequential = create_all_linear_blocks(linear_size, classes)


    def forward(self, x):

        if VERBOSE:
            ReiformInfo(x.size())

            for layer in self.conv:
                x = layer(x)
                ReiformInfo(x.size())
        else:
            x = self.conv(x)

        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

def create_deep_conv_maxpool_block(insize: int, outsize: int):
    midsize: int = (insize + outsize)//2
    return nn.Sequential(
        create_conv_block(insize, midsize//2, 3, 1, 1),
        create_conv_block(midsize//2, midsize//2, 5, 1, 2),
        create_conv_block(midsize//2, midsize, 3, 1, 1),
        create_conv_block(midsize, midsize, 5, 1, 2),
        create_conv_block(midsize, outsize, 3, 1, 1),
        nn.MaxPool2d(2, stride=2)
    )


def create_deep_conv_dilation_block(insize: int, outsize: int):
    midsize: int = (insize + outsize)//2
    return nn.Sequential(
        create_conv_block(insize, midsize, 3, 1, 1),
        create_conv_block(midsize, outsize, 3, 1, 2, d=2),
    )

class AutoResnet(nn.Module):

    def __init__(self, classes : int) -> None:
        super().__init__()

        self.features : nn.Module = torchvision.models.resnet50(pretrained=True)
        self.classifier : nn.Module = linear_block(1000, classes, relu=False, dropout=0.1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.sigmoid(x)

class DeepAutoNet(nn.Module):

    def __init__(self, insize: int, edgesize: int, classes: int) -> None:
        super().__init__()

        linear_size : int = min(4096, 400*classes)
        self.conv : nn.Sequential = create_all_conv_blocks(insize, edgesize, linear_size, 
                                            create_deep_conv_dilation_block, create_deep_conv_maxpool_block)

        self.fc : nn.Sequential = create_all_linear_blocks(linear_size, classes)


    def forward(self, x):

        if VERBOSE:
            ReiformInfo("Layer size : {}".format(x.size()))

            for layer in self.conv:
                x = layer(x)
                ReiformInfo(x.size())
        else:
            x = self.conv(x)

        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

def create_small_linear_blocks(insize : int, classes : int):

    curr_size : int = insize
    linear_blocks : List[nn.Sequential] = []

    linear_blocks.append(linear_block(curr_size, classes, dropout=0.0, relu=False))

    return nn.Sequential(*linear_blocks)


def create_small_conv_blocks(insize: int, edgesize: int, linearsize: int) -> nn.Sequential:

    max_size : int = 64**3

    curr_edge : int = edgesize
    curr_width : int = insize

    conv_blocks : List[nn.Sequential] = []

    next_width : int = max_size//(edgesize**2)

    while curr_edge * curr_edge * next_width > linearsize and curr_edge > 2:
        conv_blocks.append(
            create_conv_dilation_block(curr_width, next_width)
        )
        curr_width = next_width
        next_width //=3
        conv_blocks.append(
            create_small_conv_maxpool_block(curr_width, next_width)
        )
        curr_width = next_width
        conv_blocks.append(
            create_conv_dilation_block(curr_width, curr_width)
        )
        conv_blocks.append(
            create_small_conv_maxpool_block(curr_width, curr_width)
        )

        curr_edge //=4

    conv_blocks.append(nn.Sequential( nn.Flatten(), linear_block(curr_edge * curr_edge * curr_width,
                                    linearsize, dropout=0.1)))

    sequential = nn.Sequential(*conv_blocks)

    return sequential

class SmallAutoNet(nn.Module):

    def __init__(self, insize: int, edgesize: int, classes: int) -> None:
        super().__init__()

        linear_size : int = min(1000, 10*classes)
        self.conv : nn.Sequential = create_small_conv_blocks(insize, edgesize, linear_size)

        self.fc : nn.Sequential = create_small_linear_blocks(linear_size, classes)


    def forward(self, x):

        if VERBOSE:
            ReiformInfo(x.size())

            for layer in self.conv:
                x = layer(x)
                ReiformInfo(x.size())
        else:
            x = self.conv(x)

        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

def multiclass_model_loss(predictions : torch.Tensor, labels : torch.Tensor):
    
    loss = F.cross_entropy(predictions, labels) # compute the total loss
    return loss

def get_optimizer(model : nn.Module):
    learning_rate = 0.001
    w_d = 1e-2
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=w_d)
    return optimizer

def train_conv_net(model: nn.Module, dataloader : torch.utils.data.DataLoader, 
                   loss_function : Any, optimizer : torch.optim.Optimizer, epochs : int,
                   min_epochs : int = 10, epsilon : float = 0.00005):

  # set to training mode
  model.train()
  model.to(device)

  train_loss_avg : List[float] = []

  ReiformInfo('Training ...')
  for epoch in range(epochs):
      train_loss_avg.append(0)
      num_batches = 0
      
      for image_batch, labels, _ in dataloader:
          torch.cuda.empty_cache()
          image_batch = image_batch.to(device)
          if (num_batches % 10) == 0 and VERBOSE:
            ReiformInfo("Batch: {}".format(num_batches))
          # model reconstruction
          prediction = model(image_batch)
          labels = labels.to(device)
          # reconstruction error
          loss = loss_function(prediction, labels)
          
          # backpropagation
          optimizer.zero_grad()
          loss.backward()
          
          # one step of the optmizer (using the gradients from backpropagation)
          optimizer.step()
          
          train_loss_avg[-1] += loss.item()
          num_batches += 1
          
      train_loss_avg[-1] /= num_batches
      ReiformInfo('Epoch [%d / %d] average loss: %f' % (epoch+1, epochs, train_loss_avg[-1]))

      # Stop condition (helps with wild training times on huge datasets)
      if len(train_loss_avg) > min_epochs:
          delta_2 = train_loss_avg[-2] - train_loss_avg[-1]
          delta_1 = train_loss_avg[-3] - train_loss_avg[-2]
          if delta_1 < epsilon and delta_2 < epsilon and delta_2 > 0 and delta_1 > 0:
              break

  return model, train_loss_avg

def predict_labels(model : nn.Module, dataloader : torch.utils.data.DataLoader):

    model.eval()
    model = model.to(device)

    results : List[Tuple[str, int]] = []

    for image, label, filename in dataloader:

        image = image.to(device)

        prediction = model(image)
        prediction = prediction.to('cpu').detach().numpy()
        for pred, name in zip(prediction, filename):
            results.append((name, pred))

    return results

