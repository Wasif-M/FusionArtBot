import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_Attention(nn.Module):
    def __int__(self,channel: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)  #channels is always 32 in stable diffusion
        self.attention = SelfAttention(1,channels)
    def forward(self,x: torch.Tensor) -> torch.Tensor:

        # x:(Batch_size,channel/Features, Height, Width)
        residue = x
        n,c,h,w = x.shape
        # x:(Batch_size,channel/Features, Height, Width) -> (Batch_size,channel/Features, Height*Width)
        x=x.view(n,c,h*w)
        # x:(Batch_size,channel/Features, Height, Width) -> (Batch_size,Height*Width,CHANNEL/Features)
        x=x.transpose(-1,-2)

        #(Batch_size,Height*Width,CHANNEL/Features) -> (Batch_size,Height*Width,CHANNEL/Features) doesnt change the shape
        x=self.attention(x)

        #(Batch_size,Height*Width,CHANNEL/Features)-> (Batch_size,channel/Features, Height*Width)
        x=x.transpose(-1, -2)

        #(Batch_size,channel/Features, Height*Width) -> (Batch_size,channel/Features, Height,Width)
        x=x.view((n,c,h,w))


        x=+ residue

        return x
    

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)


        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (Batch_size, in_channels, Height, Width)

        residue =x

        x=self.group_norm1(x) #this doest not change the shape of the tensor

        x=F.silu(x) #this doest not change the shape of the tensor

        x=self.conv_1(x)    #this doest not change the shape of the tensor

        x=self.group_norm2(x) #this doest not change the shape of the tensor

        x=F.silu(x)

        x=self.conv_2(x)    #this doest not change the shape of the tensor


        return x+self.residual_layer(residue)
    


class Decoder (nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4, kernel_size=1,padding=0),

            nn.Conv2d(4,512,kernel_size=3,padding=1),

            VAE_ResidualBlock(512,512),

            VAE_Attention(512),


            VAE_ResidualBlock(512,512),



            VAE_ResidualBlock(512,512),


            VAE_ResidualBlock(512,512),


            #(Batch_size,412,Height/8,Width/8) -> (Batch_size,412,Height/8,Width/8)
            VAE_ResidualBlock(512,512),


            #(Batch_size,412,Height/8,Width/8) -> (Batch_size,512,Height/4,Width/4)
            nn.Upsample(scale_factor=2,),   #This will increase the size of the image by 2 times

            nn.Conv2d(512,512,kernel_size=3,padding=0),

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            #(Batch_size,512,Height/4,Width/4) -> (Batch_size,256,Height/2,Width/2)
            nn.Upsample(scale_factor=2,),   #This will increase the size of the image by 2 times

            nn.Conv2d(512,512,kernel_size=3,padding=0),

            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),
            #(Batch_size,256,Height/2,Width/2) -> (Batch_size,256,Height,Width)
            nn.Upsample(scale_factor=2,),   #This will increase the size of the image by 2 times


            nn.Conv2d(256.256,kernel_size=3,padding=0),


            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),

            nn.GroupNorm(32,128) ,# this will divide 128 into groups of 32 
            nn.SiLU(),

            #(Batch_size,128,Height,Width) -> (Batch_size,3(RGB),Height,Width)
            nn.Conv2d(128,3,kernel_size=3,padding=1), 

        )


def forward(self,x: torch.Tensor) -> torch.Tensor:
    # x:(Batch_size,4,Height/8,Width/8) input is latent

    x/= 0.18215

    for module in self:
        x= module(x)


    #(Batch_size,3,Height,Width)
    return x




