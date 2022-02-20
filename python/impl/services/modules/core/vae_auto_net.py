from .model_resources import *

class EncoderAutoNet(nn.Module):
    def __init__(self, in_size: int, edge_size: int, latent_size: int) -> None:
        super().__init__()

        conv_blocks : List[nn.Sequential] = []

        curr_width = in_size
        curr_edge = edge_size
        next_width = 64

        while curr_edge > max(edge_size//4, 4):
            conv_blocks.append(create_conv_block(curr_width, next_width, 4, 2, 1))
            curr_width = next_width
            next_width *=2
            curr_edge //=2

        while latent_size + 100 < curr_edge**2 * curr_width:

            if curr_edge > 4:
                curr_edge //=2
                conv_blocks.append(create_conv_block(curr_width, next_width, 4, 2, 1))
            else:
                conv_blocks.append(create_conv_block(curr_width, next_width, 3, 1, 1))
            curr_width = next_width
            next_width //=2
            
        self.conv_seq = nn.Sequential(*conv_blocks)

        self.fc_mu = nn.Linear(in_features=curr_edge**2 * curr_width, out_features=latent_size)
        self.fc_logvar = nn.Linear(in_features=curr_edge**2 * curr_width, out_features=latent_size)

    def forward(self, x):

        if VERBOSE:
            for l in self.conv_seq:
                print(x.size())
                x = l(x)
        else:
            x = self.conv_seq(x)

        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors

        x_mu = self.fc_mu(x)

        if not self.training:
            return x_mu

        x_logvar = self.fc_logvar(x)
        
        return x_mu, x_logvar

class DecoderAutoNet(nn.Module):
    def __init__(self, out_size: int, edge_size: int, latent_size: int) -> None:
        super().__init__()

        self.first_edge, self.first_width = self.get_first_sizes(latent_size)

        curr_edge  = self.first_edge
        curr_width = self.first_width

        conv_blocks : List[nn.Sequential] = []

        next_width = curr_width // 2

        while curr_edge < edge_size or next_width > out_size * 2:

            if curr_edge < edge_size:
                conv_blocks.append(create_conv_transpose_block(curr_width, next_width, 3, 1, 1))
                curr_edge *= 2
            else:
                conv_blocks.append(create_conv_block(curr_width, next_width, 3, 1, 1))

            curr_width = next_width
            if next_width > out_size * 2:
                next_width //= 2

        conv_blocks.append(create_conv_block(curr_width, out_size, 3, 1, 1, relu=False))

        self.conv_seq = nn.Sequential(*conv_blocks)

        self.fc_layer = nn.Linear(in_features=latent_size,
                                  out_features=self.first_edge**2 * self.first_width)

    def get_first_sizes(self, latent_size):
        first_edge = 4
        first_width = 8

        while first_edge**2 * first_width < latent_size * 4:

            first_width *= 2
            
            if first_edge**2 * first_width >= latent_size * 4:
                break

            first_edge *= 2

        return first_edge, first_width

    def forward(self, x):

        x = self.fc_layer(x)

        # unflatten batch of feature vectors to a batch of multi-channel 
        # feature maps
        x = x.view(x.size(0), self.first_width, self.first_edge, self.first_edge)

        x = self.conv_seq(x)

        return torch.sigmoid(x)
        

class VAEAutoNet(nn.Module):
    def __init__(self, in_size: int, edge_size: int, latent_size: int) -> None:
        super().__init__()
        self.encoder = EncoderAutoNet(in_size, edge_size, latent_size)
        self.encoder.training = True
        self.decoder = DecoderAutoNet(in_size, edge_size, latent_size)
        self.edge_size : int = edge_size

    def forward(self, x):
        
        latent_mu, latent_logvar = self.encoder(x)
        
        latent = self.latent_sample(latent_mu, latent_logvar)
        
        x_recon = self.decoder(latent)
        
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
          # reparameterization
          std = logvar.mul(0.5).exp_()
          eps = torch.empty_like(std).normal_()
          return eps.mul(std).add_(mu)
        else:
          return mu

def vae_projection_loss(recon_x, x, mu, logvar, edge_size):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    # recon_loss = F.binary_cross_entropy(recon_x.view(-1, edge_size*edge_size), x.view(-1, edge_size*edge_size), reduction='mean')
    recon_loss = nn.BCEWithLogitsLoss()(recon_x.view(-1, edge_size*edge_size), x.view(-1, edge_size*edge_size))
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Including the distance from 0 as a positive correlation in the loss drives the latent 
    # projections away from the origin to improve separation. We don't want them to blow up, 
    # so we limit the reward to 10
    dist = min(torch.linalg.vector_norm(torch.linalg.vector_norm(mu, ord=float('inf'), dim=1), ord=float('-inf')), 1)

    # print("Loss calc:")
    # print(recon_loss)
    # print(kldivergence)
    # print(dist)

    return (recon_loss + VARIATIONAL_BETA * kldivergence ) / (dist)


def train_projection_vae(vae : VAEAutoNet, dataloader : torch.utils.data.DataLoader, epochs: int, optimizer : torch.optim.Optimizer):

  # EX: optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-2)

  # set to training mode
  vae.train()
  vae.to(device)

  train_loss_avg : List[float] = []

  print('Training ...')
  for epoch in range(epochs):
      train_loss_avg.append(0)
      num_batches = 0
      
      for image_batch, _, _ in dataloader:
          torch.cuda.empty_cache()
          image_batch = image_batch.to(device)
          
          # vae reconstruction
          image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
          
          # reconstruction error
          # print(np.unique(image_batch_recon.detach().numpy()))
          # print(np.unique(image_batch.detach().numpy()))
          loss = vae_projection_loss(image_batch_recon, image_batch, latent_mu, latent_logvar, vae.edge_size)
          
          # backpropagation
          optimizer.zero_grad()
          loss.backward()
          
          # one step of the optmizer (using the gradients from backpropagation)
          optimizer.step()
          
          train_loss_avg[-1] += loss.item()
          num_batches += 1
          ITERATION =[0]
      train_loss_avg[-1] /= num_batches
      print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, epochs, train_loss_avg[-1]))
  return vae, train_loss_avg
