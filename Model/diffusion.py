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
            if isinstance(layer, UNET_AttentionBlock):
                x=layer(x,context)
            elif isinstance(layer, UNET_ResidualBlock):
                x=layer(x,time)
            else:
                x=layer(x)



class UNET_ResidualBlock(nn.Module):
    def __init__(self,in_channels: int, out_channels: int,n_time=1280): #n_time is embedding dimension
        super().__init__()
        self.group_norm1=nn.GroupNorm(32, in_channels)
        self.conv_feature=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.linear=nn.Linear(n_time,out_channels)

        self.groupnorm_merged=nn.GroupNorm(32, out_channels)
        self.conv_2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

        if in_channels==out_channels:
            self.residual_layer=nn.Identity()
        else:
            self.residual_layer=nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)

    def forward(self,feature,time):
        # features(latent):(Batch_size, in_channels, Height, Width)
        #time: (1,1280)

        residue=feature
        feature=self.group_norm1(feature)
        feature=F.silu(feature)
        feature=self.conv_feature(feature)

        time=F.silu(time)
        time= self.linear_time(time)

        merged=feature+time.unsqueeze(-1).unsqueeze(-1)

        merged=self.groupnorm_merged(merged)
        merged=F.silu(merged)
        merged=self.conv_2(merged)
        return merged+self.residual_layer(residue)
    
class UNET_AttentionBlock(nn.Module):
    def __init__(self,n_heads: int, n_embed: int,d_context: 768):
        super().__init__()
        channels=n_heads * n_embed

        self.group_norm=nn.GroupNorm(32,channels,eps=1e-6)
        self.conv_input=nn.Conv2d(channels,channels,kernel_size=1,padding=0)

        self.layernorm_1=nn.LayerNorm(channels)
        self.attention_1=SelfAttention(n_heads,channels,in_proj_bias=False)
        self.layernorm_2=nn.LayerNorm(channels)
        self.attention_2=CrossAttention(n_heads,channels,d_context,in_proj_bias=False)
        self.layernorm_3=nn.LayerNorm(channels)
        self.linear_geglu_1=nn.Linear(channels,4*channels*2)
        self.linear_geglu_2=nn.Linear(4*channels,channels)
        self.conv_output=nn.Conv2d(channels,channels,kernel_size=1,padding=0)

    def forward(self,x,context):
        # x: (Batch_size, features, Height, Width)
        # context: (Batch_size, Seq_len, Dim)

        residue_long=x
        x.self.group_norm(x)
        x=self.conv_input(x)
        n,c,h,w=x.shape
        #(Batch_size, features, Height, Width) -> (Batch_size, features, Height*Width)
        x=x.view(n,c,h*w)
        #(Batch_size, features, Height*Width) -> (Batch_size, Height*Width, features)
        x=x.transpose(-1,-2)

        # Normalization + SelfAttention with skip connection

        residue_short= x

        x=self.layernorm_1(x)
        self.attention_1
        x+=residue_short

        residue_short=x

        # Normalization + CrossAttention with skip connection

        x= self.layernorm_2(x)
        # cross attention
        self.attention_2(x,context)

        x+= residue_short
        residue_short=x

        # Normalization + FeedForward layer with GeGLu and skip connection

        x=self.layernorm_3(x)
        x,gate= self.linear_geglu_1(x).chunk(2,dim=-1)
        x=x*F.gelu(gate)
        x=self.linear_geglu_2(x)

        x+=residue_short
        #(Batch_size, Height*Width, features) -> (Batch_size, features, Height*Width)
        x=x.transpose(-1,-2)

        x=x.view(n,c,h,w)

        return self.conv_output(x) + residue_long











class UpSample(nn.Module):
    def __init__(self, in_channels: int,out_channels):
        super().__init__()
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    def forward(self,x):
        #(Batch_size,Features/channels, Height, Width) -> (Batch_size,Features/channels, Height*2, Width*2) multiplying the height and width by 2 because we are upsampling
        x=F.interpolate(x,scale_factor=2,mode='nearest')
        x=self.conv(x)



class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.ModuleList([
            # (Batch_size,4, Height/8, Width/8) 
            SwitchSequential(nn.Conv2(4,320,kernel_size=3,padding=1)),

            SwitchSequential(UNET_ResidualBlock(320,320),UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(320,320),UNET_AttentionBlock(8,40)),

            # (Batch_size,320, Height/8, Width/8) -> (Batch_size,320, Height/16, Width/16)

            SwitchSequential(nn.Conv2d(320,320,kernel_size=3,stride =2,padding=1)), #reducing the dimensionality of the image 
            
            SwitchSequential(UNET_ResidualBlock(320,640,),UNET_AttentionBlock(8,80)), # 640 is the number of channels in the image and 8 is number of heads and 80 is number of embeddings

            SwitchSequential(UNET_ResidualBlock(640,640,),UNET_AttentionBlock(8,80)),

            #(Batch_size,640, Height/16, Width/16) -> (Batch_size,640, Height/32, Width/32)

            SwitchSequential(nn.Conv2d(640,640,kernel_size=3,stride =2,padding=1)),

            SwitchSequential(UNET_ResidualBlock(640,1240),UNET_AttentionBlock(8,160)), # residual block increasing the features

            SwitchSequential(UNET_ResidualBlock(1240,1240),UNET_AttentionBlock(8,160)),
            # (Batch_size,1280, Height/32, Width/32) -> (Batch_size,1280, Height/64, Width/64)

            SwitchSequential(nn.Conv2d(1280,1280,kernel_size=3,stride =2,padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280,1280)),

                #(Batch_size,1280, Height/64, Width/64) -> (Batch_size,1280, Height/64, Width/64)
            SwitchSequential(UNET_ResidualBlock(1280,1280)),
        ])

        self.bottelneck=SwitchSequential(
            UNET_ResidualBlock(1280,1280),

            UNET_AttentionBlock(8,160),

            UNET_ResidualBlock(1280,1280),

        )

        self.decoders=nn.ModuleList([
            # we keep increasing the size of image and decreasing the number of faeatures in ths section
            # (Batch_size,2560, Height/64, Width/64) -> (Batch_size,1280, Height/64, Width/64)
            SwitchSequential(UNET_ResidualBlock(2560,1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280), UpSample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280),UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(2560,1280),UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(1920,1280),UNET_AttentionBlock(8,160),UpSample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920,640),UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(1280,640),UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(960,640),UNET_AttentionBlock(8,80),UpSample(640)),

            SwitchSequential(UNET_ResidualBlock(960,320),UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(640,320),UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(640,320),UNET_AttentionBlock(8,40)),
        ])

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.group_norm=nn.GroupNorm(32, in_channels)
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)

    def forward(self,x):
        # x: (Batch_size ,320,Height/8 ,width/8)

        x=self.group_norm(x)
        x=F.silu(x)
        x=self.conv(x)
        # (Batch_size ,4,Height/8 ,width/8)
        return x


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
    