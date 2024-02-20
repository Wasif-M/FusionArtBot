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






class CLIP(nn.Module):
    def __init__(self):
        self.embeddings = CLIPEmbeddings(49425, 768,77) # 49408 is the number of tokens in the vocabulary, 768 is the dimensionality of the text embeddings, and 77 is the dimensionality of the image embeddings

        self.layers=nn.Module([
            CLIPLayer(12,768) for i in range(12):
        ])
        self.layersnorm=nn.LayerNorm(768)


def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
    tokens=tokens.type(torch.LongTensor)

    #(Batch_size,Seq_len) -> (Batch_size,Seq_len,Dim which is 768)
    state= self.embeddings(tokens)
    for layer in self.layers:
        state=layer(state)

    #(Batch_size,Seq_len,Dim)
    output=self.layersnorm(state)

    return output


