import torch
import torch.nn as nn
from .Energy_Predictor import Energy_Predictor, Encoder, Decoder
from .swin_transformer import *
from .Unet_modules import *

class Repainter(Energy_Predictor):
    '''
    This model is SwinTransformer, but it seems doesnt work well.
    '''
    def __init__(self, input_resolution, embed_dim=96):
        super().__init__(input_resolution, embed_dim = embed_dim)
        self.head = nn.Conv2d(self.embed_dim, 3, 1)
        self.act = nn.Tanh()

    def forward(self, img:torch.Tensor, bias=None):
        b,_,_,_ = img.shape

        # Encoder
        x = self.encoder(img).permute(0,2,1)
        x = x.view(b, self.inter_dim, self.inter_H, self.inter_W)

        # Intermediate 
        inter_feature = x
        if bias is not None:
            x = x + bias
        else:
            x = x
        x = x.view(b, self.inter_dim, -1)
        x = x.permute(0,2,1)
        
        # Decoder
        x = self.decoder(x).permute(0,2,1)
        x = x.view(b, self.embed_dim, self.H, self.W) 

        x = self.head(x)    
        x = self.act(x)
        x = (x + 1) / 2

        return x, inter_feature

