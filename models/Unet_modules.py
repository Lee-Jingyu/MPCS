
import torch
import torch.nn as nn
import torch.nn.functional as F




DEBUG = False
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

def up_conv2x2(in_channels, out_channels, transpose=True):
    if transpose:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)




class DownCoXvD(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, pooling=True, norm=nn.BatchNorm2d,act=F.relu,residual=True, batch_norm=True):
        super(DownCoXvD, self).__init__()
        self.pooling = pooling
        self.residual = residual
        self.batch_norm = batch_norm
        self.bn = None
        self.pool = None
        self.conv1 = conv3x3(in_channels, out_channels)
        self.norm1 = norm(out_channels)

        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels))
        if self.batch_norm:
            self.bn = []
            for _ in range(blocks):
                self.bn.append(norm(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.ModuleList(self.conv2)
        self.act = act

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):       #下采样函数
        x1 = self.act(self.norm1(self.conv1(x)))    #relu(BatchNorm2d(Conv2d(x)))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            if self.batch_norm:
                x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2
        before_pool = x2
        if self.pooling:
            x2 = self.pool(x2)
        return x2, before_pool

class ResDown(nn.Module):
    def __init__(self, in_size, out_size, pooling=True, use_att=False):
        super(ResDown, self).__init__()
        self.model = DownCoXvD(in_size, out_size, 3, pooling=pooling)

    def forward(self, x):
        return self.model(x)

class ResUp(nn.Module):
    def __init__(self, in_size, out_size, use_att=False):
        super(ResUp, self).__init__()
        self.model = UpCoXvD(in_size, out_size, 3, use_att=use_att)

    def forward(self, x, skip_input, mask=None):
        return self.model(x,skip_input,mask)

class ResDownNew(nn.Module):
    def __init__(self, in_size, out_size, pooling=True, use_att=False):
        super(ResDownNew, self).__init__()
        self.model = DownCoXvD(in_size, out_size, 3, pooling=pooling, norm=nn.InstanceNorm2d, act=F.leaky_relu)

    def forward(self, x):
        return self.model(x)

class ResUpNew(nn.Module):
    def __init__(self, in_size, out_size, use_att=False):
        super(ResUpNew, self).__init__()
        
        self.model = UpCoXvD(in_size, out_size, 3, use_att=use_att, norm=nn.InstanceNorm2d)

    def forward(self, x, skip_input, mask=None):
        return self.model(x,skip_input,mask)

class UpCoXvD(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, residual=True,norm=nn.BatchNorm2d, act=F.relu,batch_norm=True, transpose=True,concat=True,use_att=False):
        super(UpCoXvD, self).__init__()
        self.concat = concat
        self.residual = residual
        self.batch_norm = batch_norm
        self.bn = None
        self.conv2 = []
        self.use_att = use_att
        self.up_conv = up_conv2x2(in_channels, out_channels, transpose=transpose)
        self.norm0 = norm(out_channels)
        
        if self.use_att:
            self.s2am = RASC(2 * out_channels)
        else:
            self.s2am = None

        if self.concat:
            self.conv1 = conv3x3(2 * out_channels, out_channels)
            self.norm1 = norm(out_channels , out_channels)
        else:
            self.conv1 = conv3x3(out_channels, out_channels)
            self.norm1 = norm(out_channels , out_channels)

        for _ in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels))
        if self.batch_norm:
            self.bn = []
            for _ in range(blocks):
                self.bn.append(norm(out_channels))
            self.bn = nn.ModuleList(self.bn)
        self.conv2 = nn.ModuleList(self.conv2)
        self.act = act

    def forward(self, from_up, from_down, mask=None,se=None):
        from_up = self.act(self.norm0(self.up_conv(from_up)))
        if self.concat:
            x1 = torch.cat((from_up, from_down), 1)
        else:
            if from_down is not None:
                x1 = from_up + from_down
            else:
                x1 = from_up

        if self.use_att:
            x1 = self.s2am(x1,mask)
        
        x1 = self.conv1(x1)
        x1 = self.act(self.norm1(x1))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            if self.batch_norm:
                x2 = self.bn[idx](x2)
            
            if (se is not None) and (idx == len(self.conv2) - 1): # last 
                x2 = se(x2)

            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2
        return x2

class UnetDecoderD(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, norm=nn.BatchNorm2d,act=F.relu, depth=5, blocks=1, residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True, use_att=False):
        super(UnetDecoderD, self).__init__()
        self.conv_final = None
        self.up_convs = []
        self.atts = []
        self.use_att = use_att

        outs = in_channels
        for i in range(depth-1): # depth = 1
            ins = outs
            outs = ins // 2
            # 512,256
            # 256,128
            # 128,64
            # 64,32
            up_conv = UpCoXvD(ins, outs, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat, norm=norm, act=act)
            if self.use_att:
                self.atts.append(SEBlock(outs))
            
            self.up_convs.append(up_conv)

        if is_final:
            self.conv_final = conv1x1(outs, out_channels)
        else:
            up_conv = UpCoXvD(outs, out_channels, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat,norm=norm, act=act)
            if self.use_att:
                self.atts.append(SEBlock(out_channels))

            self.up_convs.append(up_conv)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.atts = nn.ModuleList(self.atts)

        reset_params(self)

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)

    def forward(self, x, encoder_outs=None):
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            x = up_conv(x, before_pool)
            if self.use_att:
                x = self.atts[i](x)
        if self.conv_final is not None:
            x = self.conv_final(x)
        return x


class UnetDecoderDatt(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1, residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True, norm=nn.BatchNorm2d,act=F.relu, use_vm_decoder=True, 
                 use_mask_decoder=True,detach=False):
        super(UnetDecoderDatt, self).__init__()
        self.use_vm_decoder = use_vm_decoder
        self.use_mask_decoder = use_mask_decoder
        self.detach = detach
        self.conv_final = None
        self.up_convs = []
        self.im_atts = []
        self.vm_atts = []
        self.mask_atts = []
        outs = in_channels
        for i in range(depth-1): # depth = 5 [0,1,2,3]
            ins = outs
            outs = ins // 2
            # 512,256
            # 256,128
            # 128,64
            # 64,32
            up_conv = UpCoXvD(ins, outs, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat, norm=nn.BatchNorm2d,act=F.relu)
            self.up_convs.append(up_conv)
            self.im_atts.append(SEBlock(outs))
            if self.use_vm_decoder:
                self.vm_atts.append(SEBlock(outs))
            if self.use_mask_decoder:
                self.mask_atts.append(SEBlock(outs))
        if is_final:
            self.conv_final = conv1x1(outs, out_channels)
        else:
            up_conv = UpCoXvD(outs, out_channels, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat, norm=nn.BatchNorm2d,act=F.relu)
            self.up_convs.append(up_conv)
            self.im_atts.append(SEBlock(out_channels))
            if self.use_vm_decoder:
                self.vm_atts.append(SEBlock(out_channels))
            if self.use_mask_decoder:
                self.mask_atts.append(SEBlock(out_channels))

        self.up_convs = nn.ModuleList(self.up_convs)
        self.im_atts = nn.ModuleList(self.im_atts)
        if self.use_vm_decoder:
            self.vm_atts = nn.ModuleList(self.vm_atts)
        if self.use_mask_decoder:
            self.mask_atts = nn.ModuleList(self.mask_atts)

        reset_params(self)

    def forward(self, input, encoder_outs=None, mask_refine=True):
        # im branch
        # encoder_outs=shared_before_pool
        x = input
        for i, up_conv in enumerate(self.up_convs):     #上采样两次，从shared_before_pool倒数第二张开始
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            x = up_conv(x, before_pool,se=self.im_atts[i])
        x_im = x    

        x = input   
        if mask_refine:
            x_mask = []  
            for i, up_conv in enumerate(self.up_convs):
                before_pool = None
                if encoder_outs is not None:
                    before_pool = encoder_outs[-(i+2)]
                x = up_conv(x, before_pool, se = self.mask_atts[i])
                x_mask.append(x)
        elif self.use_mask_decoder:
            for i, up_conv in enumerate(self.up_convs):
                before_pool = None
                if encoder_outs is not None:
                    before_pool = encoder_outs[-(i+2)]
                if self.detach:
                    x = up_conv(x.detach(), before_pool, se = self.mask_atts[i])
                else:
                    x = up_conv(x, before_pool, se = self.mask_atts[i])
            x_mask = x  
        else:
            x_mask = None      
        
        if self.use_vm_decoder:
            x = input
            for i, up_conv in enumerate(self.up_convs):
                before_pool = None
                if encoder_outs is not None:
                    before_pool = encoder_outs[-(i+2)]
                x = up_conv(x, before_pool, se=self.vm_atts[i])
            x_vm = x
        else:
            x_vm = None

        return x_im, x_mask, x_vm

class UnetEncoderD(nn.Module):

    def __init__(self, in_channels=3, depth=5, blocks=1, start_filters=32, residual=True, batch_norm=True, norm=nn.BatchNorm2d, act=F.relu):
        super(UnetEncoderD, self).__init__()
        self.down_convs = []
        outs = None
        if type(blocks) is tuple:
            blocks = blocks[0]
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filters*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownCoXvD(ins, outs, blocks, pooling=pooling, residual=residual, batch_norm=batch_norm, norm=nn.BatchNorm2d, act=F.relu)
            self.down_convs.append(down_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        reset_params(self)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        encoder_outs = []
        for d_conv in self.down_convs:  #   4次下采样
            x, before_pool = d_conv(x)  #before_pool是池化之前的x
            encoder_outs.append(before_pool)
        return x, encoder_outs
