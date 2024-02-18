import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True,out_proj_bias=True):
        super().__int__()
        self.in_proj=nn.Linear(d_embed,d_embed*3,bias=in_proj_bias) 
        self.out_proj=nn.Linear(d_embed,d_embed,bias=out_proj_bias) # bias is not used in the original implementation
        self.n_heads=n_heads  #save the  number of heads

        self.d_embed=d_embed // n_heads #save the dimension of the embedding


    def forward(self, x: torch.Tensor, causal_mask = False):
        # x: (Batch_size, Seq_len, d_embed)
        input_shape= x.shape   # Extract the shape


        batch_size, seq_len, d_embed = input_shape  # Extract the batch size, sequence length and embedding dimension

        intermim_shape = (batch_size, seq_len, self.n_heads, self.d_heads) # Create the intermediate shape

        #(Batch_size, Seq_len, Dim) -> (Batch_size, Seq_len, Dim*3) -> 3 tensors of shape (Batch_size, Seq_len, Dim)
        q, k , v= self.in_proj(x).chunk(3, dim=-1)  # Split the input into query,key and value

        #(Batch_size, Seq_len, Dim)  -> (Batch_size, Seq_len,H, Dim/H) H represent heads -> (Batch_size, H, Seq_len, Dim/H) because we are taking transpose
        q=q.view(intermim_shape).transpose(1, 2)
        k=k.view(intermim_shape).transpose(1, 2)
        v=v.view(intermim_shape).transpose(1, 2)

        weight= q @ k.transpose(-2, -1) # (Batch_size, H, Seq_len, Seq_len)

        if causal_mask:

            # Mask the upper triangular made of 1's
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, float('-inf'))


        weight= mask.sqrt(self.d_embed) 
        weight =F.softmax(weight, dim=-1) 
        output = weight @ v  # (Batch_size, H, Seq_len, Seq_len) @ (Batch_size, H, Seq_len, Dim/H) -> (Batch_size, H, Seq_len, Dim/H)

        output=output.transpose(1, 2) # (Batch_size, H, Seq_len, Dim/H) -> (Batch_size, Seq_len, H,Dim/H)

        output = output.reshape(input_shape) # (Batch_size, Seq_len, H, Dim/H) -> (Batch_size, Seq_len, Dim)
        output = self.out_proj(output)


        #(Batch_size, Seq_len, Dim)
        return output



