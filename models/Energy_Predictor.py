from .swin_transformer import *
import torch
import torch.nn as nn
from einops import rearrange

class Encoder(SwinTransformerV2):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32], window_size=7, mlp_ratio=4, qkv_bias=True, drop_rate=0, attn_drop_rate=0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depths, num_heads, window_size, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, ape, patch_norm, use_checkpoint, pretrained_window_sizes, **kwargs)
        self.bs = nn.ParameterList()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x
    

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x

    
class Decoder(Encoder):
    '''
    Encoder的逆过程
    '''
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4, qkv_bias=True, drop_rate=0, attn_drop_rate=0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()
        
        # split image into non-overlapping patches
        self.patch_expand = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_expand.num_patches
        patches_resolution = self.patch_expand.patches_resolution
        self.patches_resolution = patches_resolution
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim // 2 ** i_layer),
                               input_resolution=(patches_resolution[0] * (2 ** i_layer),
                                                 patches_resolution[1] * (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x



class Energy_Predictor(nn.Module):
    def __init__(self, input_resolution, capacity=100, embed_dim=36, depth=4, depths=[2,2,6,2], num_heads=[4,8,16,32]):
        super().__init__()
        assert depth == len(depths), 'depth not equal with len(depths)'
        assert depth == len(num_heads), 'depth not equal with len(num_heads)'

        self.device = 'cuda:0'
        torch.cuda.set_device(self.device)
        self.capacity = capacity
        self.H,self.W = input_resolution
        self.embed_dim = embed_dim
        self.depth = depth
        self.factor = (2 ** (self.depth -1))
        self.inter_dim = self.embed_dim * self.factor
        self.inter_H = self.H // self.factor
        self.inter_W = self.W // self.factor

        self.encoder = Encoder(input_resolution, patch_size=1, in_chans=3, depths=depths, num_heads=num_heads, window_size=4, embed_dim=self.embed_dim).to(self.device)
        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.inter_dim, self.inter_H, self.inter_W, dtype=torch.float32), requires_grad=True) for i in range(capacity)
        ])
        self.mult = nn.ParameterList([nn.Parameter(torch.ones(1, self.inter_dim, self.inter_H, self.inter_W, dtype=torch.float32), requires_grad=True) for i in range(capacity)])
        self.bias_end = nn.Parameter(torch.zeros(1, 1, self.H, self.W, dtype=torch.float32), requires_grad=True)
        self.mult_end = nn.Parameter(torch.ones(1, 1, self.H, self.W, dtype=torch.float32), requires_grad=True)
        self.decoder = Decoder((self.inter_H, self.inter_W), patch_size=1, in_chans=self.inter_dim, depths=depths[::-1], num_heads=num_heads[::-1], window_size=4, embed_dim=self.inter_dim).to(self.device)
        self.head = nn.Conv2d(self.embed_dim, 1, 1)
        self.act = nn.Sigmoid()
    
    def forward(self, img:torch.Tensor, activate_layers_b=None):
        b,_,_,_ = img.shape
        x = self.encoder(img).permute(0,2,1)
        x = x.view(b, self.inter_dim, self.inter_H, self.inter_W,)
        # ---在这里考虑插入bias---
        # mask 是和bias相乘还是和x相乘，做两组实验
        if not activate_layers_b is None:
            for bi, activate_layers in enumerate(activate_layers_b):
                mask = torch.zeros(b, self.inter_dim, self.inter_H, self.inter_W, requires_grad=False, dtype=torch.float32).to(self.device)
                for i,act in enumerate(activate_layers):
                    if act == 1.:
                        mask[bi, i*4:i*4+self.factor, :, :] = 1.0 
            x = x * self.bias * mask
        else:
            new_x = x
            for mult,bias in zip(self.mult, self.bias):
                new_x = new_x * mult + bias
            x = new_x
        # --------------------
        x = x.view(b, self.inter_dim, -1)
        x = x.permute(0,2,1)
        
        x = self.decoder(x).permute(0,2,1)
        x = x.view(b, self.embed_dim, self.H, self.W) # get b

        x = self.head(x)    # get b,2,h,w
        x = self.act(x)

        return x