import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbeddings(nn.Module):
    def __init__(self,n_vocab: int ,n_embd: int ,n_tokens: int):
        super().__init__()
        self.token_embeddings=nn.Embedding(n_vocab,n_embd)
        self.position_embeddings=nn.Parameter(torch.zeros(1,n_tokens,n_embd)) #these are the parameters that rae learnt by the model during training,that tell the position of tokens

    def forward(self, tokens):
        #(Batch_size,Seq_len) -> (Batch_size,Seq_len,Dim)
        x=self.token_embeddings(tokens)
        x =+self.position_embeddings
        return x

class CLIPLayer(nn.Module):
    def __int__(self,n_head: int ,n_embd: int):
        super().__init__()
        self.layernorm_1=nn.LayerNorm(n_embd)
        self.attention=SelfAttention(n_head,n_embd)
        self.layernorm_2=nn.LayerNorm(n_embd)
        self.linear_1=nn.Linear(n_embd,4*n_embd)
        self.linear_2=nn.Linear(4*n_embd,n_embd)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #(Batch_size,Seq_len,Dim) -> (Batch_size,Seq_len,Dim)
        x=self.layernorm_1(x)
        x=self.attention(x, causal_mask=True)
        x=+ residue

        ## FEDFORWARD Layer

        residue =x
        x=self.layernorm_2(x)
        x=self.linear_1(x)
        x=x*torch.sigmoid(1.702*x)  # QuickGeLU activation function

        x=self.linear_2(x)
        x=+ residue

        return x




class CLIP(nn.Module):
    def __init__(self):
        self.embeddings = CLIPEmbeddings(49408, 768,77) # 49408 is the number of tokens in the vocabulary, 768 is the dimensionality of the text embeddings, and 77 is the dimensionality of the image embeddings

        self.layers=nn.Module ([
            CLIPLayer(12,768) for i in range(12)
        ])
        self.layernorm=nn.LayerNorm(768)


def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
    tokens=tokens.type(torch.LongTensor)

    #(Batch_size,Seq_len) -> (Batch_size,Seq_len,Dim which is 768)
    state= self.embeddings(tokens)
    for layer in self.layers:
        state=layer(state)

    #(Batch_size,Seq_len,Dim)
    output=self.layernorm(state)

    return output


