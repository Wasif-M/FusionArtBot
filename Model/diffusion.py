import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self,n_embed: int):
        super().__init__()
        self.linear_1=nn.Linear(n_embed,4*n_embed)
        self.linear_2=nn.Linear(4*n_embed,4*n_embed)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #x(1,320)
        x=self.linear_1(x)
        x=F.silu(x)
        x=self.linear_2(x)
        return x # (1,1280)


class SwitchSequential(nn.Module):
    def forward(self, x: torch.Tensor, context: torch.Tensor,time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttionBlock):
                x=layer(x,context)
            elif isinstance(layer, UNET_ResidualBlock):
                x=layer(x,time)
            else:
                x=layer(x)


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.ModuleList([
            SwitchSequential(nn.Conv2(4,320,kernel_size=3,padding=1)),

            SwitchSequential(UNET_ResidualBlock(320,320),UNET_AttionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(320,320),UNET_AttionBlock(8,40)),

            SwitchSequential(nn.Conv2d(320,320,kernel_size=3,stride =2,padding=1)), #reducing the dimensionality of the image 
            
            SwitchSequential(UNET_ResidualBlocl(320,640,),UNET_AttionBlock(8,80)), # 640 is the number of channels in the image and 8 is number of heads and 80 is number of embeddings

            SwitchSequential(UNET_ResidualBlocl(640,640,),UNET_AttionBlock(8,80)),

            SwitchSequential(nn.Conv2d(640,640,kernel_size=3,stride =2,padding=1))

            SwitchSequential(UNET_ResidualBlock(640,1240),UNET_AttionBlock(8,160)), # residual block increasing the features

            SwitchSequential(UNET_ResidualBlock(1240,1240),UNET_AttionBlock(8,160)),
        ])
class Diffusion(nn.Module):
    def self__init__(self):
        self.time_embedding = nn.Embedding(320)
        self.unet=UNET()
        self.final =UNET_OutputLayer(320,4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor,time: torch.Tensor):
        #latent : (Batch_size,4, Height/8, Width/8)
        #context : (Batch_size,Seq_len, Dim)
        #Time: (1, 320)

        time=self.time_embedding(time) # (1,320) -> (1,1280)

        # (Batch, 4 , width/8, Height/8) -> (Batch, 320, Width/8, Height/8)

        output=self.unet(latent,context,time)


        # (Batch, 320, Width/8, Height/8) -> (Batch, 4, Width/8, Height/8)
        output=self.final(output)
        # (Batch, 4, Width/8, Height/8)
        return output
    