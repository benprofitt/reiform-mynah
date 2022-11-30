from impl.services.modules.lighting_correction.lighting_resources import *

class Gaussian(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        larger = False
        if larger:
            self.kernel = torch.Tensor([[1, 4, 6, 4, 1],
                                        [4,16,24,16, 4],
                                        [6,24,36,24, 6],
                                        [4,16,24,16, 4],
                                        [1, 4, 6, 4, 1]
            ]).view(1, 1, 5, 5).repeat(3, 1, 1, 1)/256
        else:
            self.kernel = torch.Tensor([[1, 4, 4, 4, 1],
                                        [4,16,24,16, 4],
                                        [4,24,36,24, 4],
                                        [4,16,24,16, 4],
                                        [1, 4, 4, 4, 1]
            ]).view(1, 1, 5, 5).repeat(3, 1, 1, 1)/256

    def forward(self, x):
        a = nn.Conv2d(3, 3, 5, padding=2, bias=False, groups=3, padding_mode='replicate')
        with torch.no_grad():
            a.weight = nn.Parameter(self.kernel)
        a = a.to(device)
        a.requires_grad = False
        return F.avg_pool2d(a(x), 2, stride=2)

def create_gaussian_pyramid(im : torch.Tensor, levels: int) -> List[torch.Tensor]:
    pyr : List[torch.Tensor] = [im]
    g = Gaussian()
    for i in range(levels-1):
        pyr.append(g(pyr[i]))

    return pyr

def create_laplacian_pyramid(ims : List[torch.Tensor]):
    pyr : List[torch.Tensor] = []
    for i, im in enumerate(ims[:-1]):
        pyr.append(torch.sub(im, F.interpolate(ims[i+1], scale_factor=2)))
    pyr.append(ims[-1])
    return pyr

def leaky_conv_block(insize, outsize):
    return nn.Sequential(
        nn.Conv2d( insize, outsize, 3, 1, 1, bias=False, padding_mode='reflect'),
        nn.BatchNorm2d(outsize),
        nn.LeakyReLU(True),
        nn.Conv2d( outsize, outsize, 3, 1, 1, bias=False, padding_mode='reflect'),
        nn.BatchNorm2d(outsize),
        nn.LeakyReLU(True)
    )

def leaky_conv_transpose_block(insize, outsize):
    return nn.Sequential(
            # input is Z, going into a convolution
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d( insize, outsize, 3, 1, 1, dilation=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(outsize),
            nn.LeakyReLU(True)
        )
    return nn.Sequential(
            nn.ConvTranspose2d( insize, outsize, 2, 2, bias=False),
        #   nn.BatchNorm2d(outsize),
            nn.LeakyReLU(True)
        )

def maxpool_block():
    return nn.Sequential( nn.MaxPool2d(2, stride=2))

def conv_1x1_block(insize, outsize):

    return nn.Sequential(
        nn.Conv2d( insize, outsize, 1, 1, bias=False),
        nn.BatchNorm2d(outsize),
    )

def concat_layers(x, residual):
    return torch.concat((x, residual), dim=1)

def display_tensor(t1, t2):

    target1 = t1.to(device)
    target2 = t2.to(device)

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow( target1[0].cpu().permute(1, 2, 0).detach().numpy())
    axarr[1].imshow( target2[0].cpu().permute(1, 2, 0).detach().numpy())

    plt.show()

def recon_loss(x, recon_x):
    loss = torch.nn.L1Loss(reduction='mean')
    return loss(x, recon_x)

def pyramid_loss(pyr_level : List, recon_pyr_levels : List, n : int, epoch: int = 0):
    s = 0
    loss = torch.nn.L1Loss(reduction='mean')
    for l in range(0, n):
        c = 1 #2**(l)
        gt = F.interpolate(pyr_level[l], scale_factor=2)
        corr = recon_pyr_levels[len(recon_pyr_levels) - (l+1)]
        s += c * loss(gt, corr)
        # if epoch == 30:
            # display_tensor(corr, gt)
    return s

def adv_loss(x, edge_size, n):
    return -1 * torch.mean(torch.log(1 - x))

class Decoder(nn.Module):

    def __init__(self, edge_size : int, m : int, levels : int) -> None:
        # m is out channel count of layer 1
        super().__init__()
        self.edge_size = edge_size
        self.m = m
        self.levels = levels
        self.residuals : List[torch.Tensor] = []

        self.define_layers()

    def add_residual(self, res : torch.Tensor):
        self.residuals.append(res)

    def define_layers(self):
        
        layers = nn.ModuleList()
        current_width : int = 2**(self.levels - 1) * self.m
        next_width : int = current_width // 2
        for i in range(1, self.levels):
            layers.append(leaky_conv_transpose_block(current_width, next_width))
            layers.append(Concat(self))
            layers.append(leaky_conv_block(current_width, next_width))
            current_width = next_width
            next_width //= 2

        layers.append(conv_1x1_block(current_width, 3))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Concat(nn.Module):

    def __init__(self, parent : Decoder) -> None:
        super().__init__()
        self.requires_grad = False
        self.residuals = parent.residuals

    def forward(self, x):
        res = self.residuals.pop()
        return concat_layers(x, res)

class Encoder(nn.Module):

    def __init__(self, edge_size : int, m : int, levels : int) -> None:
        # m is out channel count of layer 1
        super().__init__()
        self.edge_size = edge_size
        self.m = m
        self.levels = levels

        self.decoder = Decoder(edge_size, m, levels)
        self.define_layers()

    def define_layers(self):
        self.layers = nn.ModuleList()
        current_width : int = 3
        next_width : int = self.m
        self.layers.append(leaky_conv_block(current_width, next_width))
        for i in range(1, self.levels):
            current_width = next_width
            next_width *= 2
            self.layers.append(nn.Sequential(
                    maxpool_block(),
                    leaky_conv_block(current_width, next_width)
            ))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
            self.decoder.residuals.append(x)
        self.decoder.residuals.pop()
        return self.decoder(x)
            
class Discriminator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.net = AutoNet(3, 128, 1)

    def forward(self, x):
        return self.net(x)


class LightingCorrectionNet(nn.Module):

    def __init__(self, edge_size : int) -> None:
        super().__init__()
        self.ED1 = Encoder(edge_size//8, 24, 4)
        self.t1 = leaky_conv_transpose_block(3, 3)

        self.ED2 = Encoder(edge_size//4, 24, 3)
        self.t2 = leaky_conv_transpose_block(3, 3)

        self.ED3 = Encoder(edge_size//2, 24, 3)
        self.t3 = leaky_conv_transpose_block(3, 3)

        # self.ED4 = Encoder(edge_size, 16, 3) # Original
        self.ED4 = Encoder(edge_size, 16, 2)

    def forward(self, x):

        if self.training:

            im = x
            residual : torch.Tensor
            pyramid_output : List[torch.Tensor] = []
            gaussian_pyramid = create_gaussian_pyramid(x, 4)
            laplacian_pyramid = create_laplacian_pyramid(gaussian_pyramid)

            x = laplacian_pyramid[-1]
            x = self.ED1(x)
            x = self.t1(x)
            pyramid_output.append(torch.sigmoid(torch.add(x, gaussian_pyramid[2])))
            x = torch.add(x, laplacian_pyramid[-2])
            residual = x

            x = self.ED2(x)
            x = torch.add(x, residual)
            x = self.t2(x)
            pyramid_output.append(torch.sigmoid(torch.add(x, gaussian_pyramid[1])))
            torch.add(x, laplacian_pyramid[-3])
            residual = x
            
            x = self.ED3(x)
            x = torch.add(x, residual)
            x = self.t3(x)
            pyramid_output.append(torch.sigmoid(torch.add(x, gaussian_pyramid[0])))
            torch.add(x, laplacian_pyramid[-4])
            # residual = x

            x = self.ED4(x)
            # x = torch.add(x, residual)

            return torch.sigmoid(torch.add(x, im)), pyramid_output
        
        else:
            im = x
            gaussian_pyramid = create_gaussian_pyramid(x, 4)
            laplacian_pyramid = create_laplacian_pyramid(gaussian_pyramid)

            x = laplacian_pyramid[-1]
            x = self.ED1(x)
            x = self.t1(x)
            x = torch.add(x, laplacian_pyramid[-2])
            residual = x

            x = self.ED2(x)
            x = torch.add(x, residual)
            x = self.t2(x)
            torch.add(x, laplacian_pyramid[-3])
            residual = x
            
            x = self.ED3(x)
            x = torch.add(x, residual)
            x = self.t3(x)
            torch.add(x, laplacian_pyramid[-4])
            # residual = x

            x = self.ED4(x)
            # x = torch.add(x, residual)

            return torch.sigmoid(torch.add(x, im))

def train_lighting_correction(model : LightingCorrectionNet, dataloader : torch.utils.data.DataLoader,
                              edge_size : int, epochs : int, 
                              optimizer : torch.optim.Optimizer, epoch_start=0):
  
  discriminator = Discriminator()
  discriminator.train()
  discriminator = discriminator.to(device)
  disc_optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001, weight_decay=1e-2)
  def disc_loss_func(gt, target, batch_size):
    loss = F.binary_cross_entropy(gt, torch.zeros([batch_size, 1], device=device)) if target == 0 \
            else F.binary_cross_entropy(gt, torch.ones( [batch_size, 1], device=device))
    return loss

  def loss_func(pyramid, recon_pyramid, recon, target, disc_recon, epoch):
      pyr_loss = pyramid_loss(pyramid, recon_pyramid, len(pyramid), epoch)
      rec_loss = recon_loss(target, recon)
      dis_loss = 0
      if epoch >= 30 and epoch < 35:
        dis_loss = adv_loss(disc_recon, edge_size, 1)
        

      return pyr_loss + rec_loss + dis_loss/2

  # set to training mode
  model.train()
  model.to(device)

  train_loss_avg : List[float] = []

  ReiformInfo('Training ...')
  for epoch in range(epoch_start, epochs):
      train_loss_avg.append(0)
      num_batches = 0
      
      for image_batch, target_image_batch, labels in dataloader:
          with torch.autograd.set_detect_anomaly(True):
            empty_mem_cache()

            image_batch = image_batch.to(device)
            target_image_batch = target_image_batch.to(device)
            
            # vae reconstruction

            # Discriminator loss
            discriminator.zero_grad()
            disc_optimizer.zero_grad()
            disc_loss_real = disc_loss_func(discriminator(target_image_batch), 0, target_image_batch.size()[0])
            disc_loss_real.backward()

            recon_x, pyr_out = model(image_batch)
            disc_recon = discriminator(recon_x.detach())
            disc_loss_fake = disc_loss_func(disc_recon, 1, target_image_batch.size()[0])
            disc_loss_fake.backward()
            
            # reconstruction loss
            model.zero_grad()
            gauss_pyr = create_gaussian_pyramid(target_image_batch, 4)
            loss = loss_func(gauss_pyr[1:], pyr_out, recon_x, target_image_batch, discriminator(recon_x), epoch)
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            disc_optimizer.step()
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            num_batches += 1
            
      train_loss_avg[-1] /= num_batches
      ReiformInfo('Epoch [{} / {}] avg loss: {}'.format(epoch+1, epochs, round(train_loss_avg[-1], 4)))

      # A checkpoint system for model training 
      m_path : str = "checkpoint.pt"
      torch.save(model.state_dict(), m_path)
      model.train()
  return model, train_loss_avg

def train_lighting_correction_with_validation(model : LightingCorrectionNet, dataloader : torch.utils.data.DataLoader,
                              val_dataloader : torch.utils.data.DataLoader, edge_size : int,
                              epochs : int, optimizer : torch.optim.Optimizer, epoch_start=0):
  
  discriminator = Discriminator()
  discriminator.train()
  discriminator = discriminator.to(device)
  disc_optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001, weight_decay=1e-2)
  def disc_loss_func(gt, target, batch_size):
    loss = F.binary_cross_entropy(gt, torch.zeros([batch_size, 1], device=device)) if target == 0 \
            else F.binary_cross_entropy(gt, torch.ones( [batch_size, 1], device=device))
    return loss

  def loss_func(pyramid, recon_pyramid, recon, target, disc_recon, epoch):
      pyr_loss = pyramid_loss(pyramid, recon_pyramid, len(pyramid), epoch)
      rec_loss = recon_loss(target, recon)
      dis_loss = 0
      if epoch >= 30 and epoch < 35:
        dis_loss = adv_loss(disc_recon, edge_size, 1)
        

      return pyr_loss + rec_loss + dis_loss/2

  # set to training mode
  model.train()
  model.to(device)

  train_loss_avg : List[float] = []

  ReiformInfo('Training ...')
  for epoch in range(epoch_start, epochs):
      train_loss_avg.append(0)
      num_batches = 0
      
      for image_batch, target_image_batch, labels in dataloader:
          with torch.autograd.set_detect_anomaly(True):
            empty_mem_cache()

            image_batch = image_batch.to(device)
            target_image_batch = target_image_batch.to(device)
            

            # vae reconstruction

            # Discriminator loss
            discriminator.zero_grad()
            disc_optimizer.zero_grad()
            disc_loss_real = disc_loss_func(discriminator(target_image_batch), 0, target_image_batch.size()[0])
            disc_loss_real.backward()

            recon_x, pyr_out = model(image_batch)
            disc_recon = discriminator(recon_x.detach())
            disc_loss_fake = disc_loss_func(disc_recon, 1, target_image_batch.size()[0])
            disc_loss_fake.backward()
            
            # reconstruction loss
            model.zero_grad()
            gauss_pyr = create_gaussian_pyramid(target_image_batch, 4)
            loss = loss_func(gauss_pyr[1:], pyr_out, recon_x, target_image_batch, discriminator(recon_x), epoch)
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            disc_optimizer.step()
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            num_batches += 1
            
      train_loss_avg[-1] /= num_batches
      ReiformInfo('Epoch [{} / {}] avg loss: {}'.format(epoch+1, epochs, round(train_loss_avg[-1], 4)))

      # TODO: Add a checkpoint system for all model training (and reusable model training code)
      m_path : str = "checkpoint.pt"
      torch.save(model.state_dict(), m_path)
      eval_model(val_dataloader, model)
      model.train()
  return model, train_loss_avg


def eval_model(dataloader : torch.utils.data.DataLoader, model : LightingCorrectionNet):

    model = model.to(device)
    model.eval()

    total_loss : int = 0
    total : int = 0

    labels : Dict[int, int] = {}

    prediction_losses : Dict[int, List[float]] = {}

    for image_batch, target_image_batch, label in dataloader:
        target_image_batch = target_image_batch.to(device)
        image_batch = image_batch.to(device)
        recon = model(image_batch)
        for i in range(len(label)):

            loss = recon_loss(target_image_batch[i], recon[i]).cpu().item()
            lab = int(label[i])
            if lab not in prediction_losses:
                prediction_losses[lab] = []
            if lab not in labels:
                labels[lab] = 0
            prediction_losses[lab].append(loss)
            labels[lab] = labels[lab] + 1
            total_loss += loss
            total += 1
            

    ReiformInfo("Avg Loss = {}".format(round(total_loss/total, 3)))
    for k in labels:
        ReiformInfo("Class {} = {}".format(k, round(sum(prediction_losses[k])/labels[k], 3)))

