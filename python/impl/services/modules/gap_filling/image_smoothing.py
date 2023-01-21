import io
import os, sys
import requests
import PIL
import numpy as np

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from dall_e          import map_pixels, unmap_pixels, load_model

TARGET_IMAGE_SIZE = 256

class ImageSmoother(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ImageSmoother, cls).__new__(cls)
            cls.dev, cls.enc, cls.dec = load_models()
        return cls.instance

    def forward(cls, x):

        smooth_fcn = create_smoothing_function(cls.dev, cls.enc, cls.dec)
        return smooth_fcn(x)

def vocab_to_r1024(z):
    vector = np.zeros(1024)
    for v in torch.nonzero(z):
        vector[v[2] * 32 + v[3]] = v[1] #/8192
    return vector

def r1024_to_vocab(vector):
    z = torch.zeros((1, 8192, 32, 32))
    for i, value in enumerate(vector):
        col = i%32
        row = int((i-col)/32)
        z[0, int(value), row, col] = 1
        
    return z

def preprocess(img):
    s = min(img.size)
    
    img = T.ToTensor()(img)
    if s < TARGET_IMAGE_SIZE:
        img = T.Resize((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))(img)

    img = TF.center_crop(img, output_size=2 * [TARGET_IMAGE_SIZE])
    img = torch.unsqueeze(img, 0)
    return map_pixels(img)

def load_models():
    # This can be changed to a GPU, e.g. 'cuda:0'.
    dev = torch.device('cuda')

    # For faster load times, download these files locally and use the local paths instead.
    base_path = "/home/ben/Code/filling_data_gaps/dall_e_vae_models"
    # enc = load_model("https://cdn.openai.com/dall-e/encoder.pkl", dev)
    # dec = load_model("https://cdn.openai.com/dall-e/decoder.pkl", dev)
    enc = load_model("{}/encoder.pkl".format(base_path), dev)
    dec = load_model("{}/decoder.pkl".format(base_path), dev)
    
    return dev, enc, dec

def create_smoothing_function(dev, enc, dec):


    def encode_image_32x32(img):
        # 1
        img = preprocess(img)

        img = img.to(dev)
        z_logits = enc(img)
        z = torch.argmax(z_logits, axis=1)
        return z

    def encode_32x32_to_vocab(vector):
        z = F.one_hot(vector, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float()
        return z

    def vocab_to_image(vocab_vector):
        x_stats = dec(vocab_vector).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        x_rec = T.ToPILImage(mode='RGB')(x_rec[0])

        return x_rec


    def smooth_image(image):
        smooth_image = vocab_to_image(encode_32x32_to_vocab(encode_image_32x32(image)))
        return smooth_image

    return smooth_image

if __name__ == "__main__":
    ImageSmoother()