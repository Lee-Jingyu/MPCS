import torch
import torch.nn as nn

class recall_loss(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, pred, gt):
        assert pred.ndim == 4, "wrong input shape"
        b,c,h,w = pred.shape
        assert c == 2, "channel must be 2"
        
        pred1 = pred[:,1,:,:]
        gt1 = gt[:,1,:,:]

        inter = torch.sum(pred1 * gt1)
        gt_sum = torch.sum(gt1)
        recall = inter / gt_sum
        return 1 - recall