import torch
from torch import  nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Decoder(nn.Sequential): ## each module will use to reduce the dimension of data and same time increase the number of data
    def __init__(self):
        super.__init__(
            #(Batch_size,Channel, Height, Width) -> (Batch_size, 128, Height, Width) output will in same size because we added the padding
            nn.Conv2d(3,128,kernel_size=3,padding=1),

            #(Batch_size,128, Height, Width) -> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128,128),# this indicates how many input and output channel we have #combination of both normalization and convolutions


            #(Batch_size,128, Height, Width) -> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(256,128),


            #(Batch_size,128, Height, Width) -> (Batch_size, 128, Height/2, Width/2)
            nn.Conv2d(128,128,kernel_size=3,padding=0, stride=2),  #   will reduce the size of the image by 2

            #(Batch_size,128, Height, Width) -> (Batch_size, 128, Height/2, Width/2)
            VAE_ResidualBlock(128,256),


            #(Batch_size,256, Height, Width) -> (Batch_size, 128, Height/2, Width/2)
            VAE_ResidualBlock(256,256),


        )