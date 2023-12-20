import torch
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import swin
from einops import rearrange,repeat

class PPMBASIC(nn.Module):
    def __init__(self, in_dim,mid, reduction_dim, bins):
        super(PPMBASIC, self).__init__()
        self.features = []
        self.comp=nn.Conv2d(in_dim, mid, kernel_size=1, bias=True)
        self.features = nn.ModuleList([self._make_ppm(mid, reduction_dim, bin) for bin in bins])
        self.bins = bins

    def _make_ppm(self, in_dim, reduction_dim, bin):
        return nn.Sequential(
            nn.AvgPool2d(bin, stride=bin, padding=(bin-1)//2, count_include_pad=False),
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=True),
            nn.GroupNorm(16,reduction_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        ppm_out = [x]
        x=self.comp(x)
        b,c,h,w=x.shape
        ppm_out += [F.interpolate( feature(x),(h,w),mode='bilinear')    for feature in self.features]
        ppm_out = torch.cat(ppm_out, dim=1)
        return ppm_out

class BasicmattingNEO(nn.Module):
    def __init__(self):
        super(BasicmattingNEO, self).__init__()
        trans = swin.SwinTransformer(pretrain_img_size=224,
                                     embed_dim=96,
                                     depths=[2, 2, 6,2],
                                     num_heads=[3, 6, 12,24],
                                     window_size=7,
                                     ape=False,
                                     drop_path_rate=0.2,
                                     patch_norm=True,
                                     use_checkpoint=False)
        trans.patch_embed.proj = nn.Conv2d(32,96,3,2,1)
        self.start_conv = nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.PReLU(32),nn.Conv2d(32,32,3,1,1),nn.PReLU(32))
        self.trans=trans
        self.ppm=PPMBASIC(768,256,128,[3,7,11,15])
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=768+512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True))
        self.conv2=nn.Sequential(nn.Conv2d(in_channels=256+384, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),  )
        self.conv3=nn.Sequential(nn.Conv2d(in_channels=256+192, out_channels=192, kernel_size=1, stride=1, padding=0, bias=True),   )
        self.conv4=nn.Sequential(nn.Conv2d(in_channels=192+96, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),       )
        self.tran0=swin.BasicLayer(256,3,8,7,drop_path=[0.12,0.11,0.10])
        self.tran1=swin.BasicLayer(256,3,8,7,drop_path=[0.09,0.08,0.07])
        self.tran2=swin.BasicLayer(192,3,6,7,drop_path=[0.06,0.05,0.04])
        self.tran3=swin.BasicLayer(128,3,4,7,drop_path=[0.03,0.02,0.01])
        self.conv5=nn.Sequential(nn.Conv2d(in_channels=128+32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),nn.PReLU(64),nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),nn.PReLU(64),nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),nn.PReLU(32))
        self.convtri=nn.Sequential(nn.Conv2d(in_channels=128+32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),nn.GroupNorm(16,64),nn.GELU(),nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),nn.GroupNorm(8,64),nn.GELU(),nn.Dropout(0.2),nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True))
        self.convo=nn.Sequential(nn.Conv2d(in_channels=32+3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),nn.PReLU(32),nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),nn.PReLU(16),nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True))
        self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self,x):
        inputs=x
        x_=self.start_conv(inputs)
        x1,x2,x3,x4=self.trans(x_)
        x4=self.ppm(x4)
        X4=self.conv1(x4)
        wh,ww=X4.shape[2],X4.shape[3]
        X4=rearrange(X4,'b c h w -> b (h w) c')
        X4, _, _, _, _, _=self.tran0(X4,wh,ww)
        X4=rearrange(X4,'b (h w) c -> b c h w',h=wh,w=ww)
        X3=self.up(X4)
        X3=torch.cat((x3,X3),1)
        X3=self.conv2(X3)
        wh,ww=X3.shape[2],X3.shape[3]
        X3=rearrange(X3,'b c h w -> b (h w) c')
        X3, _, _, _, _, _=self.tran1(X3,wh,ww)
        X3=rearrange(X3,'b (h w) c -> b c h w',h=wh,w=ww)
        X2=self.up(X3)
        X2=torch.cat((x2,X2),1)
        X2=self.conv3(X2)
        wh,ww=X2.shape[2],X2.shape[3]
        X2=rearrange(X2,'b c h w -> b (h w) c')
        X2, _, _, _, _, _=self.tran2(X2,wh,ww)
        X2=rearrange(X2,'b (h w) c -> b c h w',h=wh,w=ww)
        X1=self.up(X2)
        X1=torch.cat((x1,X1),1)
        X1=self.conv4(X1)
        wh,ww=X1.shape[2],X1.shape[3]
        X1=rearrange(X1,'b c h w -> b (h w) c')
        X1, _, _, _, _, _=self.tran3(X1,wh,ww)
        X1=rearrange(X1,'b (h w) c -> b c h w',h=wh,w=ww)
        X0=self.up(X1)
        X0=torch.cat((x_,X0),1)
        tri=self.convtri(X0)
        tri=self.up(tri)
        X0=self.conv5(X0)
        X=self.up(X0)
        X=torch.cat((inputs,X),1)
        alpha=self.convo(X)
        return alpha,tri

if __name__ == '__main__':
    a=BasicmattingNEO().cuda()
    a.eval()
    c=torch.randn(1,3,640,640).cuda()
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(a, c)
    print(flops.total()/1e9)
    c=torch.randn(4,3,640,640).cuda()
    torch.backends.cudnn.benchmark=True
    import time
    while True:
        z=time.time()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for i in range(10):
                    b=a(c)
            torch.cuda.synchronize()
        print(40./( time.time()-z))