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
    