from impl.services.modules.lighting_correction.lighting_resources import *

class LightingDetector(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.c1 = create_conv_block(3, 128, 3, 1, 1)  # 256 x 256
        self.c2 = create_conv_block(128, 128, 3, 1, 2, 2)
        self.c3 = create_conv_maxpool_block(128, 128) # 128 x 128
        self.c4a = create_conv_block(128, 128, 3, 1, 2, 2)
        self.c4b = create_conv_block(128, 128, 3, 1, 1)
        self.c5 = create_conv_maxpool_block(256, 128) # 64 x 64
        self.c6 = create_conv_maxpool_block(128, 128) # 32 x 32
        self.c7a = create_conv_block(128, 32, 3, 1, 1)
        self.c7b = create_conv_block(128, 32, 3, 1, 3, 3)
        self.c8 = create_conv_maxpool_block(64, 32) # 16 x 16
        self.c9 = create_conv_maxpool_block(32, 16) # 8 x 8

        self.conv1 = nn.Sequential(*[self.c1,
                                    self.c2,
                                    self.c3])

        self.fc1a = linear_block(1024, 256)
        self.fc1b = linear_block(1024, 256)

        self.fc2a = linear_block(256, 64)
        self.fc2b = linear_block(256, 64)

        self.fc3 = linear_block(128, 3, relu=False)

    def forward(self, x):
        x = self.conv1(x)
        xa = self.c4a(x)
        xb = self.c4b(x)
        x = torch.cat((xa, xb), dim=1)

        x = self.c5(x)
        x = self.c6(x)
        xa = self.c7a(x)
        xb = self.c7b(x)
        x = torch.cat((xa, xb), dim=1)

        x = self.c8(x)
        x = self.c9(x)

        x = torch.flatten(x, start_dim=1)

        xa = self.fc1a(x)
        xb = self.fc1b(x)

        xa = self.fc2a(xa)
        xb = self.fc2b(xb)

        x = torch.cat((xa, xb), dim=1)
        x = self.fc3(x)

        return torch.sigmoid(x)


def sparse_maxpool_block(in_size, out_size):
    return nn.Sequential(
        nn.MaxPool2d(2, 2),
        create_conv_block(in_size, out_size, 3, 1, 1)
    )

class LightingDetectorSparse(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.c1 = create_conv_block(3, 64, 3, 1, 1)  # 128
        self.c2 = create_conv_block(64, 96, 3, 1, 2, 2)
        self.c3 = sparse_maxpool_block(96, 128) # 64
        self.c4a = create_conv_block(32, 32, 3, 1, 2, 2)
        self.c4b = create_conv_block(32, 32, 3, 1, 1)
        self.c4c = create_conv_block(32, 32, 5, 1, 2)
        self.c4d = create_conv_block(32, 32, 3, 1, 3, 3)
        # self.c5 = sparse_maxpool_block(128, 64) # 64
        self.c6 = sparse_maxpool_block(128, 64) # 32
        self.c7a = create_conv_block(32, 32, 3, 1, 1)
        self.c7b = create_conv_block(32, 32, 3, 1, 3, 3)
        self.c8 = sparse_maxpool_block(64, 32) # 16 x 16
        self.c9 = sparse_maxpool_block(32, 16) # 8 x 8

        self.conv1 = nn.Sequential(*[self.c1,
                                     self.c2,
                                    self.c3])

        # 1024 * (256 + 256) -> (512 * 256) + (512 * 256)

        self.fc1a = linear_block(128, 32)
        self.fc1b = linear_block(128, 32)
        self.fc1c = linear_block(128, 32)
        self.fc1d = linear_block(128, 32)
        self.fc1e = linear_block(128, 32)
        self.fc1f = linear_block(128, 32)
        self.fc1g = linear_block(128, 32)
        self.fc1h = linear_block(128, 32)

        self.fc2a = linear_block(128, 64)
        self.fc2b = linear_block(128, 64)

        # self.fc3 = linear_block(128, 3, dropout=0.0, relu=False)
        self.fc3 = linear_block(128, 2, dropout=0.0, relu=False)

    def forward(self, x):
        x = self.conv1(x)

        xa, xb, xc, xd = torch.split(x, 32, dim=1)
        xa = self.c4a(xa)
        xb = self.c4b(xb)
        xc = self.c4c(xc)
        xd = self.c4d(xd)
        x = torch.cat((xa, xb, xc, xd), dim=1)

        # x = self.c5(x)
        x = self.c6(x)
        xa, xb = torch.split(x, [32, 32], dim=1)
        xa = self.c7a(xa)
        xb = self.c7b(xb)
        x = torch.cat((xa, xb), dim=1)

        x = self.c8(x)
        x = self.c9(x)

        x = torch.flatten(x, start_dim=1)

        xa, xb, xc, xd, xe, xf, xg, xh = torch.split(x, 128, dim=1)

        xa = self.fc1a(xa)
        xb = self.fc1b(xb)
        xc = self.fc1c(xc)
        xd = self.fc1d(xd)
        xe = self.fc1e(xe)
        xf = self.fc1f(xf)
        xg = self.fc1g(xg)
        xh = self.fc1h(xh)

        xa = torch.cat((xa, xb, xc, xd), dim=1)
        xb = torch.cat((xe, xf, xg, xh), dim=1)

        xa = self.fc2a(xa)
        xb = self.fc2b(xb)

        x = torch.cat((xa, xb), dim=1)
        x = self.fc3(x)

        return torch.sigmoid(x)



class Inception_ForLightingDetection(nn.Module):

    def __init__(self):
        super().__init__()
        self.main = torchvision.models.inception_v3(pretrained=True, transform_input=False)
        self.aux_lin = linear_block(1000, 2, dropout=0.0, relu=False)
        self.final_lin = linear_block(1000, 2, dropout=0.0, relu=False)

    def forward(self, x):
        x = self.main(x)
        if self.training:
            aux_x = self.aux_lin(x.aux_logits)
            x = self.final_lin(x.logits)
            return aux_x, x
        else:
            x = self.final_lin(x)
            return x


class Inception_ForLightingDetection_Both(nn.Module):

    def __init__(self):
        super().__init__()
        self.main = torchvision.models.inception_v3(pretrained=True, transform_input=False)
        self.aux_lin = linear_block(1000, 3, dropout=0.0, relu=False)
        self.final_lin = linear_block(1000, 3, dropout=0.0, relu=False)

    def forward(self, x):
        x = self.main(x)
        if self.training:
            aux_x = self.aux_lin(x.aux_logits)
            x = self.final_lin(x.logits)
            return aux_x, x
        else:
            x = self.final_lin(x)
            return x

def eval_model(dataloader, model):
    model = model.to(device)
    model.eval()

    correct : int = 0
    total : int = 0

    labels = {}
    labels_correct = {}

    predictions = {}
    predictions_correct = {}

    for image, label in dataloader:
        image = image.to(device)
        pred = model(image)
        for i in range(len(label)):    
            val = int(torch.argmax(pred[i]))
            lab = int(label[i])
            if val not in predictions:
                predictions[val] = 0
                predictions_correct[val] = 0
            if lab not in labels:
                labels[lab] = 0
                labels_correct[lab] = 0
            predictions[val] = predictions[val] + 1
            labels[lab] = labels[lab] + 1
            total += 1
            if val == lab:
                correct += 1
                predictions_correct[val] = predictions_correct[val] + 1
                labels_correct[lab] = labels_correct[lab] + 1

    ReiformInfo("Accuracy: {}/{} = {}".format(correct, total, round(correct/total, 3)))
    for k in labels:
        ReiformInfo("Class {}: {}/{} = {}".format(k, labels_correct[k], labels[k], round(labels_correct[k]/labels[k], 3)))

def train_inception_detection(model : LightingDetector, dataloader : torch.utils.data.DataLoader, val_dataloader : torch.utils.data.DataLoader, epochs: int, optimizer : torch.optim.Optimizer):

  # EX: optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-2)
  def loss_func(pred, target):
    return nn.CrossEntropyLoss()(pred, target)


  # set to training mode
  model.train()
  model.to(device)

  train_loss_avg : List[float] = []

  ReiformInfo('Training ...')
  for epoch in range(epochs):
      train_loss_avg.append(0)
      num_batches = 0
      
      for image_batch, labels in dataloader:
          
          if image_batch.size()[0] == 1:
            continue
          torch.cuda.empty_cache()
          image_batch = image_batch.to(device)
          labels = labels.to(device)
          
          # preds
          aux_pred, predictions = model(image_batch)
          
          # loss
          pred_loss = loss_func(predictions, labels)
          aux_loss = loss_func(aux_pred, labels)

          loss = pred_loss + 0.4*aux_loss
          
          # backpropagation
          optimizer.zero_grad()
          loss.backward()
          
          # one step of the optmizer (using the gradients from backpropagation)
          optimizer.step()
          
          train_loss_avg[-1] += loss.item()
          num_batches += 1
          
      train_loss_avg[-1] /= num_batches
      
      ReiformInfo('Epoch [%d / %d] avg loss: %f' % (epoch+1, epochs, round(train_loss_avg[-1], 3)))
      eval_model(val_dataloader, model)
      model.train()
  return model, train_loss_avg

def train_detection(model : LightingDetectorSparse, dataloader : torch.utils.data.DataLoader, 
                    val_dataloader : torch.utils.data.DataLoader, 
                    epochs: int, optimizer : torch.optim.Optimizer):

  # EX: optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-2)
  def loss_func(pred, target):
    return nn.CrossEntropyLoss()(pred, target)


  # set to training mode
  model.train()
  model.to(device)

  train_loss_avg : List[float] = []

  ReiformInfo('Training ...')
  for epoch in range(epochs):
      train_loss_avg.append(0)
      num_batches = 0
      
      for image_batch, labels in dataloader:
          torch.cuda.empty_cache()
          image_batch = image_batch.to(device)
          labels = labels.to(device)
          
          # vae reconstruction
          predictions = model(image_batch)
          
          # reconstruction error
          loss = loss_func(predictions, labels)
          
          # backpropagation
          optimizer.zero_grad()
          loss.backward()
          
          # one step of the optmizer (using the gradients from backpropagation)
          optimizer.step()
          
          train_loss_avg[-1] += loss.item()
          num_batches += 1
          ITERATION =[0]
      train_loss_avg[-1] /= num_batches
      ReiformInfo('Epoch [%d / %d] avg loss: %f' % (epoch+1, epochs, round(train_loss_avg[-1], 3)))
      eval_model(val_dataloader, model)
      model.train()
  return model, train_loss_avg