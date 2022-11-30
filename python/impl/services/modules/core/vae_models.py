from .resources import *
from .model_resources import *
    
def vae_loss(recon_x, x, mu, logvar):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 32*32), x.view(-1, 32*32), reduction='sum')
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    dist = min(torch.linalg.vector_norm(torch.linalg.vector_norm(mu, ord=float('inf'), dim=1), ord=float('-inf')), 10)

    return (recon_loss + VARIATIONAL_BETA * kldivergence ) / (dist)



class linear_Encoder(nn.Module):
    def __init__(self, in_size: int , latent_size : int):
        super(linear_Encoder, self).__init__()
        c = 64

        self.fc1 = linear_block(in_size, 8*c)
        self.fc2 = linear_block(8*c, 8*c)
        self.fc3 = linear_block(8*c, 4*c)
        self.fc4 = linear_block(4*c, 2*c)
        self.fc5 = linear_block(2*c, c)

        self.fc_mu = nn.Linear(in_features=c, out_features=latent_size)
        self.fc_logvar = nn.Linear(in_features=c, out_features=latent_size)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class linear_Decoder(nn.Module):
    def __init__(self, out_size: int, latent_size : int):
        super(linear_Decoder, self).__init__()
        c = 64

        self.fc6 = linear_block(latent_size, c)
        self.fc5 = linear_block(c, 2*c)
        self.fc4 = linear_block(2*c, 4*c)
        self.fc3 = linear_block(4*c, 8*c)
        self.fc2 = linear_block(8*c, 8*c)
        self.fc1 = nn.Linear(in_features=8*c, out_features=out_size)
 
    def forward(self, x):
        x = self.fc6(x)
        x = self.fc5(x)
        x = self.fc4(x)
        x = self.fc3(x)
        x = self.fc2(x)
        x = self.fc1(x)
        return torch.sigmoid(x)

class linear_VAE(nn.Module):
    def __init__(self, in_size: int, latent_size : int):
        super(linear_VAE, self).__init__()
        self.encoder = linear_Encoder(in_size, latent_size)
        self.decoder = linear_Decoder(in_size, latent_size)
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
          # the reparameterization trick
          std = logvar.mul(0.5).exp_()
          eps = torch.empty_like(std).normal_()
          return eps.mul(std).add_(mu)
        else:
          return mu

def linear_vae_loss(recon_x, x, mu, logvar, clss: Any):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution

    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # classification_loss = np.sum(np.absolute(torch.argmax(mu, axis=1).to("cpu").detach().numpy() - clss.to("cpu").detach().numpy())) #type: ignore
    class_arr = np.array([[0, 1] if x == 1 else [1, 0] for x in clss.to("cpu").detach().numpy()])
    classification_loss = np.sum(np.absolute(mu.to("cpu").detach().numpy() - class_arr)) #type: ignore

    return (recon_loss + VARIATIONAL_BETA * kldivergence ) * classification_loss * classification_loss * classification_loss


def train_linear_vae(vae : nn.Module, dataloader : torch.utils.data.DataLoader, optimizer : torch.optim.Optimizer, epochs : int = 20):
    vae.to(device)

    # set to training mode
    vae.train()

    train_loss_avg : List[float] = []

    ReiformInfo('Training ...')
    for epoch in range(epochs):
        train_loss_avg.append(0)
        num_batches = 0
        if epoch % 5 == 0:
            ITERATION =[1]
        for projections, classes, filenames in dataloader:
            empty_mem_cache()
            image_batch = projections.to(device)
            # vae reconstruction
            image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
            
            # reconstruction error
            loss = linear_vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar, classes)
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()
            
            train_loss_avg[-1] += loss.item()
            num_batches += 1
            ITERATION =[0]
        train_loss_avg[-1] /= num_batches
        ReiformInfo('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, epochs, train_loss_avg[-1]))
    return vae, train_loss_avg

def train_vae(vae : nn.Module, dataloader : Any, optimizer : torch.optim.Optimizer, epochs: int):

  # set to training mode
  vae.to(device)
  vae.train()

  train_loss_avg : List[float] = []

  ReiformInfo('Training ...')
  for epoch in range(epochs):
      train_loss_avg.append(0)
      num_batches = 0
      if epoch % 5 == 0:
        ITERATION =[1]
      for image_batch, _ in dataloader:
          empty_mem_cache()
          image_batch = image_batch.to(device)
          
          # vae reconstruction
          image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
          
          # reconstruction error
          loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
          
          # backpropagation
          optimizer.zero_grad()
          loss.backward()
          
          # one step of the optmizer (using the gradients from backpropagation)
          optimizer.step()
          
          train_loss_avg[-1] += loss.item()
          num_batches += 1
          ITERATION =[0]
      train_loss_avg[-1] /= num_batches
      ReiformInfo('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, epochs, train_loss_avg[-1]))
  return vae, train_loss_avg

