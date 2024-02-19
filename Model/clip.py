import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIP(nn.Module):
    def __init__(self):
        self.embeddings = CLIPEmbeddings(49408, 768,77) # 49408 is the number of tokens in the vocabulary, 768 is the dimensionality of the text embeddings, and 77 is the dimensionality of the image embeddings

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
    output=self.layersnorm(state)

    return output


