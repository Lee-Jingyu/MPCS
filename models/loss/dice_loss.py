import torch
import torch.nn as nn

class dice_loss(nn.Module):
    def __init__(self, smooth=1e-5, channel_weights=[1,1]) -> None:
        super().__init__()
        self.smooth = smooth
        self.channel_weights = channel_weights

    def forward(self, prediction, target):
        '''
        Input shape:  (C,H,W).
        先计算每个channel的dice loss,如channel=2时,各channel的loss为a,b。
        最终loss的公式为 (a*b)/(a+b)。
        '''
        assert prediction.ndim == 4, "wrong input shape"
        b,c,h,w = prediction.shape
        
        sum = 0
        prd = 1
        for p,g in zip(prediction, target):
            for p1, g1, w in zip(p,g,self.channel_weights):
                intersection = torch.sum(p1 * g1)
                union = torch.sum(p1) + torch.sum(g1)
                dice =  1 - (2. * intersection + self.smooth) / (union + self.smooth)
                sum += dice * w
            # prd *= dice
        # loss = prd / sum
        return sum/(c*b)