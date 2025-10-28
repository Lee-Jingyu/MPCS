import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import math
import numbers


from torch import nn, cuda
from torch.autograd import Variable

class BasicLearningBlock(nn.Module):
    """docstring for BasicLearningBlock"""
    def __init__(self,channel):
        super(BasicLearningBlock, self).__init__()
        self.rconv1 = nn.Conv2d(channel,channel*2,3,padding=1,bias=False)
        self.rbn1 = nn.BatchNorm2d(channel*2)
        self.rconv2 = nn.Conv2d(channel*2,channel,3,padding=1,bias=False)
        self.rbn2 = nn.BatchNorm2d(channel)

    def forward(self,feature):
        return F.elu(self.rbn2(self.rconv2(F.elu(self.rbn1(self.rconv1(feature)))))) 

class MaskLearningBlock(nn.Module):
    def __init__(self, channel):
        super(MaskLearningBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, 1, 7,padding=3)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1,1,1)
        self.bn2 = nn.BatchNorm2d(1)
        self.conv3 = nn.Conv2d(1,1,1)
        self.bn3 = nn.BatchNorm2d(1)
        self.up = None

    def forward(self, feature, mask_H):
        '''
        input params:
        1. feature(B,C,w,h) or (B,C,w,h), anyway the same in img_size(256,256)
        2. mask_H: height of raw mask

        output:
        1.refine(B,1,W,H), an offset of mask
        '''
        _,_,_,h = feature.size()
        ratio = mask_H / h
        #初始化up模块
        if self.up == None:
            if ratio == 8:
                self.up = nn.ModuleList([
                    nn.ConvTranspose2d(1,1,kernel_size=2,stride=2,padding=0),
                    nn.ConvTranspose2d(1,1,kernel_size=2,stride=2,padding=0),
                    nn.ConvTranspose2d(1,1,kernel_size=2,stride=2,padding=0)
                ])
            elif ratio == 4:
                self.up = nn.ModuleList([
                    nn.ConvTranspose2d(1,1,kernel_size=2,stride=2,padding=0),
                    nn.ConvTranspose2d(1,1,kernel_size=2,stride=2,padding=0)
                ])
            elif ratio == 2:
                self.up = nn.ModuleList([
                    nn.ConvTranspose2d(1,1,kernel_size=2,stride=2,padding=0)
                ])
            else:
                self.up = nn.ModuleList([
                    nn.Identity()
                ])
            self.up.to('cuda')

        #计算mask refine
        refine = torch.tanh(self.bn2(self.conv2(torch.tanh(self.bn1(self.conv1(feature))))))

        for up in self.up:
            refine = up(refine)
        refine = torch.tanh(self.bn3(self.conv3(refine)))
            
        return refine
    


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

class ChannelPool(nn.Module):
    def __init__(self,types):
        super(ChannelPool, self).__init__()
        if types == 'avg': 
            self.poolingx = nn.AdaptiveAvgPool1d(1)
        elif types == 'max':
            self.poolingx = nn.AdaptiveMaxPool1d(1)
        else:
            raise 'inner error'

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n,c,w*h).permute(0,2,1) 
        pooled =  self.poolingx(input)# b,w*h,c ->  b,w*h,1
        _, _, c = pooled.size()
        return pooled.view(n,c,w,h)



class SEBlock(nn.Module):
    """docstring for SEBlock"""
    def __init__(self, channel,reducation=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reducation),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reducation,channel),
            nn.Sigmoid())
        
    def forward(self,x):
        b,c,w,h = x.size()
        y1 = self.avg_pool(x).view(b,c)
        y = self.fc(y1).view(b,c,1,1)
        return x*y



class GlobalAttentionModule(nn.Module):
    """docstring for GlobalAttentionModule"""
    def __init__(self, channel,reducation=16):
        super(GlobalAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel*2,channel//reducation),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reducation,channel),
            nn.Sigmoid())
        
    def forward(self,x):
        b,c,w,h = x.size()
        y1 = self.avg_pool(x).view(b,c)
        y2 = self.max_pool(x).view(b,c)
        y = self.fc(torch.cat([y1,y2],1)).view(b,c,1,1)
        return x*y
    
class FC_Layer(nn.Module):
    def __init__(self, in_channel, out_channel, reducation):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channel,in_channel//reducation),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel//reducation,out_channel),
            nn.Sigmoid())
    
    def forward(self, x):
        return self.fc(x)
    
class MaskedAvgPool2d(nn.Module):
    #  不好用
    def __init__(self) -> None:
        super().__init__()

    def forward(self, feature, mask):
        b,c,h,w = feature.size()
        mask_sum = torch.sum(mask, dim=(2, 3), keepdim=True).expand(b,c,1,1)  # 在指定维度上求和
        expanded_mask = mask.expand(b,c,h,w)
        feature_sum = torch.sum(feature * expanded_mask, dim=(2, 3), keepdim=True)  # 元素相乘后求和
        mask_sum = torch.clamp(mask_sum, min=1e-6)  # 防止除以零
        result = feature_sum / mask_sum  # 相除操作
        
        return result
        


class InteractiveAttentionModule(nn.Module):
    def __init__(self, channel,reducation=16):
        super().__init__()
        self.fg_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fg_max_pool = nn.AdaptiveMaxPool2d(1)
        self.bg_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.bg_max_pool = nn.AdaptiveMaxPool2d(1)
        self.mixed_fc = FC_Layer(channel*4,channel,reducation)
        self.fc1 = FC_Layer(channel*3, channel, reducation)
        self.fc2 = FC_Layer(channel*3, channel, reducation)
        
    def forward(self, feature, mask):
        b,c,w,h = feature.size()
        _,_,mw,_ = mask.size()

        # if w != mw:
        #     mask = torch.round(F.avg_pool2d(mask,2,stride=mw//w))
        # else:
        #     mask = torch.round(mask)
        if w != mw:
            mask = F.avg_pool2d(mask,2,stride=mw//w)
        else:
            mask = mask
        reverse_mask = -1 * (mask-1)


        fg = (feature * mask)
        bg = (feature * reverse_mask)        

        b,c,w,h = fg.size()
        f1 = self.fg_avg_pool(fg).view(b,c)
        f2 = self.fg_max_pool(fg).view(b,c)
        b1 = self.fg_avg_pool(bg).view(b,c)
        b2 = self.fg_max_pool(bg).view(b,c)
        mixed_feature = self.mixed_fc(torch.cat([f1,f2,b1,b2],1)).view(b,c)
        fg_feature = self.fc1(torch.cat([f1,f2,mixed_feature],1)).view(b,c,1,1)
        bg_feature = self.fc2(torch.cat([b1,b2,mixed_feature],1)).view(b,c,1,1)

        return fg*fg_feature + bg*bg_feature
    

class InteractiveAttentionModule_v1(nn.Module):
    def __init__(self, channel,reducation=16):
        super().__init__()
        self.fg_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fg_max_pool = nn.AdaptiveMaxPool2d(1)
        self.bg_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.bg_max_pool = nn.AdaptiveMaxPool2d(1)
        self.mixed_fc = FC_Layer(channel*4,channel,reducation)
        self.fc1 = FC_Layer(channel*2, channel, reducation)
        self.fc2 = FC_Layer(channel*2, channel, reducation)
        
    def forward(self, feature, mask):
        b,c,w,h = feature.size()
        _,_,mw,_ = mask.size()

        if w != mw:
            mask = torch.round(F.avg_pool2d(mask,2,stride=mw//w))
        else:
            mask = torch.round(mask)
        reverse_mask = -1 * (mask-1)


        fg = (feature * mask)
        bg = (feature * reverse_mask)        

        b,c,w,h = fg.size()
        f1 = self.fg_avg_pool(feature,mask).view(b,c)
        f2 = self.fg_max_pool(fg).view(b,c)
        b1 = self.fg_avg_pool(feature,mask).view(b,c)
        b2 = self.fg_max_pool(bg).view(b,c)

        cross_attention = f1 * b1   # (b,1,c,c)
        fg_feature = (f2 * cross_attention).view(b,c,1,1)
        bg_feature = (b2 * cross_attention).view(b,c,1,1)

        return fg*fg_feature + bg*bg_feature

class InteractiveAttentionModule_v2(InteractiveAttentionModule_v1):
    def __init__(self, channel, reducation=16):
        super().__init__(channel, reducation)
    
    def forward(self, feature, mask):
        b,c,w,h = feature.size()
        _,_,mw,_ = mask.size()

        if w != mw:
            mask = torch.round(F.avg_pool2d(mask,2,stride=mw//w))
        else:
            mask = torch.round(mask)
        reverse_mask = -1 * (mask-1)


        fg = (feature * mask)
        bg = (feature * reverse_mask)        

        b,c,w,h = fg.size()
        f1 = self.fg_avg_pool(fg).view(b,1,c,1)
        b1 = self.fg_avg_pool(bg).view(b,1,1,c)
        f2 = self.fg_max_pool(fg).view(b,1,1,c)
        b2 = self.fg_max_pool(bg).view(b,1,1,c)

        cross_attention = torch.matmul(f1, b1)   # (b,1,c,c)
        fg_feature = torch.matmul(f2,cross_attention).view(b,c,1,1)
        bg_feature = torch.matmul(b2,cross_attention).view(b,c,1,1)

        return fg*fg_feature + bg*bg_feature

class SpatialAttentionModule(nn.Module):
    """docstring for SpatialAttentionModule"""
    def __init__(self, channel,reducation=16):
        super(SpatialAttentionModule, self).__init__()
        self.avg_pool = ChannelPool('avg')
        self.max_pool = ChannelPool('max')
        self.fc = nn.Sequential(
            nn.Conv2d(2,reducation,7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(reducation,1,7,stride=1,padding=3),
            nn.Sigmoid())
        
    def forward(self,x):
        b,c,w,h = x.size()
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y = self.fc(torch.cat([y1,y2],1))
        yr = 1-y
        return y,yr



class GlobalAttentionModuleJustSigmoid(nn.Module):
    """docstring for GlobalAttentionModule"""
    def __init__(self, channel,reducation=16):
        super(GlobalAttentionModuleJustSigmoid, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel*2,channel//reducation),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reducation,channel),
            nn.Sigmoid())
        
    def forward(self,x):
        b,c,w,h = x.size()
        y1 = self.avg_pool(x).view(b,c)
        y2 = self.max_pool(x).view(b,c)
        y = self.fc(torch.cat([y1,y2],1)).view(b,c,1,1)
        return y



class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicBlock, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPoolX(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPoolX()
        self.spatial = BasicBlock(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


