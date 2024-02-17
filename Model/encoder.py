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
            VAE_ResidualBlock(128,128),


            #(Batch_size,128, Height, Width) -> (Batch_size, 128, Height/2, Width/2)
            nn.Conv2d(128,128,kernel_size=3,padding=0, stride=2),  #   will reduce the size of the image by 2

            #(Batch_size,128, Height, Width) -> (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128,256),


            #(Batch_size,256, Height/4, Width/4) -> (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256,256),

            #(Batch_size,256, Height, Width) -> (Batch_size, 256, Height/4, Width/4)
            nn.Conv2d(256,256,kernel_size=3,padding=0, stride=2),  

            #(Batch_size,256, Height/4, Width/4) -> (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256,512),

            #(Batch_size,256, Height/4, Width/4) -> (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512,512),

            #(Batch_size,512, Height/4, Width/4) -> (Batch_size, 512, Height/8, Width/8)
            nn.Conv2d(512,512,kernel_size=3,padding=0, stride=2), 

            VAE_ResidualBlock(512,512),


            VAE_ResidualBlock(512,512),

            #(Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512,512),


            VAE_AttentionBlock(512),

            
            #(Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512,512),
            #(Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)

            nn.GroupNorm(32,512),

            #(Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            nn.SiLU(), # Swish activation function


            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 8, Height/8, Width/8)
            nn.Conv2d(512,8,kernel_size=3,padding=1),

            # (Batch_size, 8, Height/8, Width/8) -> (Batch_size, 8, Height/8, Width/8)
            nn.Conv2d(8,8,kernel_size=1,padding=0), 


        )

    def forward(self, x: torch.Tensor , noise: torch.Tensor) -> torch.Tensor:      # x is image we want to encode
        # x: (Batch_size, channel, Height, Width)
        #noise: (Batch_size, out_channel, Height/8, Width/8)

        for module in self:
            if getattr(module,'stride',None)==(2,2):
                #(padding_left,padding_right,padding_top,padding_bottom)
                x=F.pad(x,(0,1,0,1)) # padding the image to make the size of the image same after the convolution
            x=module(x)


        # (Batch_size, 8, Height/8, Width/8) -> (Batch_size, 4, Height/8, Width/8)
        mean, log_variance = x.chunk(2,dim=1) # split the tensor into two parts

        # (Batch_size, 4, Height/8, Width/8) -> (Batch_size, 4, Height/8, Width/8)
        log_variance=torch.clamp(log_variance,min=-30,max=20)   # clamp the value of log_variance between -30 and 20

        # (Batch_size, 4, Height/8, Width/8) -> (Batch_size, 4, Height/8, Width/8)

        variance=log_variance.exp() # exponential of log_variance

        # (Batch_size, 4, Height/8, Width/8) -> (Batch_size, 4, Height/8, Width/8)

        stdev=variance.sqrt()


        # Z=N(0,1) -> N(mean, variance)=x*stdev+mean

        x=stdev*noise+mean

        #scale the output by constant

        x=x*0.18215         # this constant is used by the paper publisher
        return x
