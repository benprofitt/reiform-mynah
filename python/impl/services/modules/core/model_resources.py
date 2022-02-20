from .resources import *
from .reiform_imageclassificationdataset import *

VARIATIONAL_BETA = 0.0000001
VAE_PROJECTION_TRAINING_EPOCHS = 100

def create_conv_transpose_block(insize, outsize, k, s, p, d=1, relu=True):
  if relu:
    return nn.Sequential(
              # input is Z, going into a convolution
              nn.UpsamplingNearest2d(scale_factor=2),
              nn.Conv2d( insize, outsize, k, s, p, dilation=d, bias=False),
              nn.BatchNorm2d(outsize),
              nn.ReLU(True)
          )
  else:
    return nn.Sequential(
              # input is Z, going into a convolution
              nn.UpsamplingNearest2d(scale_factor=2),
              nn.Conv2d( insize, outsize, k, s, p, dilation=d, bias=False),
              nn.BatchNorm2d(outsize)
          )

def create_conv_block(insize, outsize, k, s, p, d=1, relu=True):

    if relu:
        return nn.Sequential(
                    nn.Conv2d( insize, outsize, k, s, p, d, bias=False),
                    nn.BatchNorm2d(outsize),
                    nn.ReLU(True)
        )
    else:
        return nn.Sequential(
                    nn.Conv2d( insize, outsize, k, s, p, d, bias=False),
                    nn.BatchNorm2d(outsize),
        )


def linear_block(insize, outsize, dropout=0.4, relu=True):
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