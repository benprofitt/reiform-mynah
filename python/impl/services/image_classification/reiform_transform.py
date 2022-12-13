from datetime import datetime
import os
from typing import List, Dict
import torch
from torch import nn
import torchvision
from torchvision import transforms
from resources import *
from torchvision.utils import save_image

sys.path.insert(1, '/home/ben/Code/filling_data_gaps/BigGAN-PyTorch')

device = 'cuda'
EMBEDDING_DIM_SIZE = 120

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

class DatasetWithEmbeddings(torch.utils.data.Dataset):

    def __init__(self, dataset: ReiformICDataSet, edge_size: int = 128) -> None:
        super().__init__()

        self.dataset = dataset

        models_path = "python/models/"
        embedding_models_path = "{}/{}".format(models_path, EMBEDDING_MODEL_NAME)
        create_dataset_embedding(dataset, embedding_models_path)

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.transform : torchvision.transforms.Compose = transforms.Compose([
                transforms.Resize(edge_size),
                transforms.CenterCrop(edge_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=mean, std=std)
            ])

        self.chosen_class = dataset.classes()[0]
        # self.files_and_embeddings = [(f, f.get_projection(PROJECTION_LABEL_REDUCED_EMBEDDING)) for _, f in self.dataset.get_items(self.chosen_class)]
        self.files_and_embeddings = []
        for c in self.dataset.classes():
            for _, f in self.dataset.get_items(c):
                self.files_and_embeddings.append((f, f.get_projection(PROJECTION_LABEL_REDUCED_EMBEDDING)))

    def __len__(self):
        return len(self.files_and_embeddings)

    def __getitem__(self, idx: int):

        file, emb = self.files_and_embeddings[idx]

        image = self.dataset.read_image(file)

        return self.transform(image), torch.tensor(emb).to(device)

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

def create_laplacian_loss(x, x_recon):
    gaussian_pyramid = create_gaussian_pyramid(x, 4)
    laplacian_pyramid = create_laplacian_pyramid(gaussian_pyramid)

    gaussian_pyramid = create_gaussian_pyramid(x_recon, 4)
    laplacian_pyramid_recon = create_laplacian_pyramid(gaussian_pyramid)

    total_loss = 0
    # loss = torch.nn.L1Loss(reduction='mean')
    loss = torch.nn.MSELoss()
    for i in range(len(laplacian_pyramid)):
        total_loss += loss(laplacian_pyramid[i], laplacian_pyramid_recon[i])

    return(total_loss/len(x_recon))

def loss_latent_transform(im_o, G, v_c):
    y = torch.zeros(v_c.size()[0], dtype=int)
    for i in range(v_c.size()[0]):
        y[i] = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = y.to(device)

    new_ims = G(v_c, G.shared(y))

    loss1 = torch.nn.MSELoss()(im_o, new_ims)
    loss2 = torch.nn.L1Loss(reduction='mean')(im_o, new_ims)

    return loss1 + create_laplacian_loss(im_o, new_ims)

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

def loss_func(pyramid, recon_pyramid, recon, target, epoch):
    pyr_loss = pyramid_loss(pyramid, recon_pyramid, len(pyramid), epoch)
    rec_loss = recon_loss(target, recon)
    
    return pyr_loss + rec_loss

def save_new_image(ims, embs, G):

    y = torch.zeros(embs.size()[0], dtype=int).to(device)

    start = round(time.time())

    save_image(G(embs, G.shared(y))[0], 'python/data/results/{}_new.png'.format(start))
    save_image(ims[0], 'python/data/results/{}_ori.png'.format(start))

    # std_embs = torch.normal(0, 1, embs.size()).to(device)

    # save_image(G(std_embs, G.shared(y))[1], 'python/data/results/normal_{}.png'.format(start+1))

def get_optimizer_linear(model : nn.Module):
    learning_rate = 0.1
    w_d = 1e-3
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=w_d)
    return optimizer

def train_transformation(G, dataloader, epochs):

    model = TransformModel()
    optimizer = get_optimizer_linear(model)

    model.train()
    model.to(device)

    train_loss_avg : List[float] = []

    for epoch in range(epochs):
        train_loss_avg.append(0)
        num_batches = 0

        for image_batch, embeddings in dataloader:
            torch.cuda.empty_cache()
            embeddings = embeddings.to(device)
            
            # model reconstruction
            prediction = model(embeddings)
            image_batch = image_batch.to(device)
            # reconstruction error
            loss = loss_latent_transform(image_batch, G, prediction)
            
            if num_batches == 0:
                save_new_image(image_batch, prediction, G)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()
            
            train_loss_avg[-1] += loss.item()
            num_batches += 1

        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average loss: %f' % (epoch+1, epochs, train_loss_avg[-1]))

    return model

def linear_block(insize, outsize, dropout=0.02, relu=True):
    if relu:
        return nn.Sequential(
            nn.Linear(in_features=insize, out_features=outsize),
            nn.Dropout(p=dropout),
            nn.ReLU(True)
        )
    else:
        return nn.Sequential(
            nn.Linear(in_features=insize, out_features=outsize),
            nn.Dropout(p=dropout)
        )

def standardize(t):
    std = torch.std(t)
    mean = torch.mean(t)

    # print("Mean: {}\nStd: {}".format(mean, std))

    return (t - mean)/std

class TransformModel(nn.Module):

    def __init__(self) -> None:
       super().__init__()
       self.latent_origin = EMBEDDING_DIM_SIZE
       self.latent_target = 120
       c = 120
       self.fc1 = linear_block(self.latent_origin, c)
       self.fc2 = linear_block(c, 2*c)
       self.fc2_1 = linear_block(2*c, 2*c)
       self.fc2_2 = linear_block(2*c, 2*c)
       self.fc3 = linear_block(2*c, c)
       self.fc4 = linear_block(c, self.latent_target, relu=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc2_1(x)
        # x = self.fc2_2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return standardize(x)
        # return torch.tanh(x)*3

def join_strings(base_string, strings):
  return base_string.join([item for item in strings if item])


def load_weights(G, D, state_dict, weights_root, experiment_name, 
                 name_suffix=None, G_ema=None, strict=True, load_optim=True):
  root = '/'.join([weights_root, experiment_name])
  if name_suffix:
    print('Loading %s weights from %s...' % (name_suffix, root))
  else:
    print('Loading weights from %s...' % root)
  if G is not None:
    G.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['G', name_suffix]))),
      strict=strict)
    if load_optim:
      G.optim.load_state_dict(
        torch.load('%s/%s.pth' % (root, join_strings('_', ['G_optim', name_suffix]))))
  if D is not None:
    D.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['D', name_suffix]))),
      strict=strict)
    if load_optim:
      D.optim.load_state_dict(
        torch.load('%s/%s.pth' % (root, join_strings('_', ['D_optim', name_suffix]))))
  # Load state dict
  for item in state_dict:
    state_dict[item] = torch.load('%s/%s.pth' % (root, join_strings('_', ['state_dict', name_suffix])))[item]
  if G_ema is not None:
    G_ema.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['G_ema', name_suffix]))),
      strict=strict)

def load_generator_model(config_path):
    with open(config_path, 'r') as fp:
        config = json.load(fp)
        state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}
        experiment_name = config['experiment_name']

        config['model'] = 'BigGAN'
        model = __import__(config['model'])

        G = model.Generator(**config).cuda()

        load_weights(G if not (config['use_ema']) else None, None, state_dict, 
                     config['weights_root'], experiment_name, config['load_weights'],
                     G if config['ema'] and config['use_ema'] else None,
                     strict=False, load_optim=False)

        return G

def main():

    dataset_path = sys.argv[1]

    G = load_generator_model(sys.argv[2])

    dataset = dataset_from_path(dataset_path)
    image_data = DatasetWithEmbeddings(dataset)
    batch_size = 16
    shuffle = True
    
    dataloader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    epochs = 50

    G.eval()

    model = train_transformation(G, dataloader, epochs)

if __name__ == "__main__":
    main()