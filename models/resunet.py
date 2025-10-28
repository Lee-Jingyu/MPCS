
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Unet_modules import *

from PIL import Image

from torchvision.utils import save_image

s2am_registry = {}

def decoder_register(key):
    def decorator(obj):
        s2am_registry[key] = obj
        return obj
    return decorator

@decoder_register('vms2am_baseline')
class VMSingleS2AM_baseline(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, down=ResDown, up=ResUp, ngf=32):
        super().__init__()

        self.down1 = down(in_channels, ngf)
        self.down2 = down(ngf, ngf*2)
        self.down3 = down(ngf*2, ngf*4)
        self.down4 = down(ngf*4, ngf*8)
        self.down5 = down(ngf*8, ngf*16, pooling=False)

        self.up1 = up(ngf*16, ngf*8)

        self.up2 = up(ngf*8, ngf*4)
        
        self.up3 = up(ngf*4, ngf*2)

        self.up4 = up(ngf*2, ngf*1)

        self.im = nn.Conv2d(ngf, 3, 1)


    def forward(self, input):
        # U-Net generator with skip connections from encoder to decoder
        x, d1 = self.down1(input) # 128,256
        x, d2 = self.down2(x) # 64,128
        x, d3 = self.down3(x) # 32,64
        x, d4 = self.down4(x) # 16,32
        x,  _ = self.down5(x) # 16,_

        x = self.up1(x, d4) # 32

        x = self.up2(x, d3) # 64

        x = self.up3(x, d2) # 128

        x = self.up4(x, d1) # 256

        im = self.im(x)
        return im


class UnetVMS2AMv4(nn.Module):
    def __init__(self, in_channels=3, depth=5, shared_depth=0, use_vm_decoder=False, blocks=1,use_MBLA=True,
                 out_channels_image=3, out_channels_mask=1, start_filters=32, residual=True, batch_norm=True,
                 transpose=True, concat=True, transfer_data=True, long_skip=False, 
                 s2am='unet', use_coarser=True,no_stage2=False,
                 mask_refine=False,use_mask_decoder=True,detach = False,IAM_depths=[2,2,2,2]):
        super(UnetVMS2AMv4, self).__init__()
        self.detach = detach
        self.mask_refine = mask_refine
        self.use_MBLA = use_MBLA
        self.transfer_data = transfer_data
        self.shared = shared_depth
        self.optimizer_encoder,  self.optimizer_image, self.optimizer_vm = None, None, None
        self.optimizer_mask, self.optimizer_shared = None, None
        if type(blocks) is not tuple:
            blocks = (blocks, blocks, blocks, blocks, blocks)
        if not transfer_data:
            concat = False
        
        self.encoder = UnetEncoderD(in_channels=in_channels, depth=depth, blocks=blocks[0],
                                    start_filters=start_filters, residual=residual, 
                                    batch_norm=batch_norm,
                                    norm=nn.InstanceNorm2d,act=F.leaky_relu)
        self.image_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                          out_channels=out_channels_image, depth=depth - shared_depth,
                                          blocks=blocks[1], residual=residual, batch_norm=batch_norm,
                                          transpose=transpose, concat=concat,norm=nn.InstanceNorm2d)
        self.mask_decoder = None
        if use_mask_decoder:
            self.mask_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                            out_channels=out_channels_mask, depth=depth - shared_depth,
                                            blocks=blocks[2], residual=residual, batch_norm=batch_norm,
                                            transpose=transpose, concat=concat,norm=nn.InstanceNorm2d)
        
        self.vm_decoder = None
        if use_vm_decoder:
            self.vm_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                           out_channels=out_channels_image, depth=depth - shared_depth,
                                           blocks=blocks[3], residual=residual, batch_norm=batch_norm,
                                           transpose=transpose, concat=concat,norm=nn.InstanceNorm2d)
        self.shared_decoder = None
        self.use_coarser = use_coarser
        self.long_skip = long_skip
        self.no_stage2 = no_stage2
        if self.shared != 0:
            self._forward = self.shared_forward
            self.shared_decoder = UnetDecoderDatt(in_channels=start_filters * 2 ** (depth - 1),
                                               out_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                               depth=shared_depth, blocks=blocks[4], residual=residual,
                                               batch_norm=batch_norm, transpose=transpose, concat=concat,
                                               is_final=False,norm=nn.InstanceNorm2d, use_vm_decoder=use_vm_decoder,
                                               use_mask_decoder=use_mask_decoder,detach=detach
                                               )
        
        
        self.s2am = VMSingleS2AM_baseline(3,down=ResDownNew,up=ResUpNew)   
        self.s2am_type = 'vms2am_baseline'    


    def set_optimizers(self, lr=0.001):
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.optimizer_image = torch.optim.Adam(self.image_decoder.parameters(), lr=lr)
        self.optimizer_mask = torch.optim.Adam(self.mask_decoder.parameters(), lr=lr)
        if self.s2am:
            self.optimizer_s2am = torch.optim.Adam(self.s2am.parameters(), lr=lr)

        if self.vm_decoder is not None:
            self.optimizer_vm = torch.optim.Adam(self.vm_decoder.parameters(), lr=lr)
        if self.shared != 0:
            self.optimizer_shared = torch.optim.Adam(self.shared_decoder.parameters(), lr=lr)

    def zero_grad_all(self):
        self.optimizer_encoder.zero_grad()
        self.optimizer_image.zero_grad()
        self.optimizer_mask.zero_grad()
        if self.s2am:
            self.optimizer_s2am.zero_grad()
        if self.vm_decoder is not None:
            self.optimizer_vm.zero_grad()
        if self.shared != 0:
            self.optimizer_shared.zero_grad()

    def step_all(self):     
        self.optimizer_encoder.step()
        self.optimizer_image.step()
        self.optimizer_mask.step()
        if self.s2am:
            self.optimizer_s2am.step()
        if self.vm_decoder is not None:
            self.optimizer_vm.step()
        if self.shared != 0:
            self.optimizer_shared.step()

    def step_optimizer_image(self):
        self.optimizer_image.step()

    def __call__(self, synthesized):
        return self._forward(synthesized)

    def forward(self, synthesized):
        return self._forward(synthesized)   

    def shared_forward(self, synthesized):
        image_code, before_pool = self.encoder(synthesized) 

        if self.transfer_data:
            shared_before_pool = before_pool[- self.shared - 1:]
            unshared_before_pool = before_pool[: - self.shared]
        else:
            before_pool = None
            shared_before_pool = None
            unshared_before_pool = None
        
        im, mask, vm = self.shared_decoder(image_code, shared_before_pool, mask_refine=self.mask_refine)
        
        reconstructed_image = self.image_decoder(im, unshared_before_pool)
        act = nn.Sigmoid()
        coarser = act(reconstructed_image)
       
        return coarser
