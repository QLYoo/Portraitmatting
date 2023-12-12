import torch
from torchvision.models import resnet34
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F

__all__ = ['ResNet',  'resnet34', ]

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.baseconv=nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                                              bias=True),nn.PReLU(16))

        self.baseconv1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(32),nn.ReLU())
        self.baseconv2=nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        x=self.baseconv(x)
        x1=self.baseconv1(x)
        x2=self.baseconv2(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)
        return x,x1,x2,x3,x4,x5,x6

def _resnet(
        arch,
        block,
        layers,
        pretrained,
):
    model = ResNet(block, layers,)
    if pretrained:
        model.load_state_dict(torch.load('resnet34-b627a593.pth') ,strict=False)
    return model


def resnet34(pretrained: bool = True,) :
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained)


class PPM(nn.Module):
    def __init__(self, in_dim,mid, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        self.comp=nn.Conv2d(in_dim, mid, kernel_size=1, bias=True)
        self.features = nn.ModuleList([self._make_ppm(mid, reduction_dim, [bin,bin]) for bin in bins])
        self.bins = bins

    def _make_ppm(self, in_dim, reduction_dim, bin):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(bin),
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        ppm_out = [x]
        x=self.comp(x)
        b,c,h,w=x.shape
        ppm_out += [F.interpolate( feature(x),(h,w),mode='bilinear')    for feature in self.features]
        ppm_out = torch.cat(ppm_out, dim=1)
        return ppm_out


class ResBlock(nn.Module):
    def __init__(self, inc, midc):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inc, midc, kernel_size=1, stride=1, padding=0, bias=True)
        self.gn1=nn.BatchNorm2d(midc)
        self.conv2 = nn.Conv2d(midc, midc, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn2=nn.BatchNorm2d(midc)
        self.conv3=nn.Conv2d(midc,inc,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self,x):
        x_=x
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x+x_
        x = self.relu(x)
        return x


class Basicmatting(nn.Module):
    def __init__(self):
        super(Basicmatting, self).__init__()
        self.resnet=resnet34(pretrained=False)
        self.ppm=PPM(512,256,128,[1,2,3,6])
        self.dec0=nn.Sequential(nn.Conv2d(1024,256,1,1,0),ResBlock(256,192),ResBlock(256,192))
        self.dec1=nn.Sequential(nn.Conv2d(512,192,1,1,0),ResBlock(192,128),ResBlock(192,128))
        self.dec2=nn.Sequential(nn.Conv2d(320,128,1,1,0),ResBlock(128,96),ResBlock(128,96))
        self.dec3=nn.Sequential(nn.Conv2d(192,96,1,1,0),ResBlock(96,64),ResBlock(96,64))
        self.dec4=nn.Sequential(nn.Conv2d(128,48,1,1,0),ResBlock(48,32),ResBlock(48,32))
        self.deco=nn.Sequential(nn.Conv2d(128,128,1,1,0),nn.BatchNorm2d(128),nn.ReLU(),nn.Conv2d(128,128,1,1,0),nn.BatchNorm2d(128),nn.ReLU(),nn.Dropout(0.2),nn.Conv2d(128,3,1,1,0),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False))
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.dec5=nn.Sequential(nn.Conv2d(48+16,24,3,1,1),nn.PReLU(24),nn.Conv2d(24,1,3,1,1))
    def forward(self, x):
        x,x1,x2,x3,x4,x5,x6=self.resnet(x)
        outfeat=self.ppm(x6)
        outfeat=self.dec0(outfeat)
        outfeat=self.up(outfeat)
        outfeat=torch.cat((outfeat,x5),1)
        outfeat=self.dec1(outfeat)
        outfeat=self.up(outfeat)
        outfeat=torch.cat((outfeat,x4),1)
        outfeat=self.dec2(outfeat)
        outfeat=self.up(outfeat)
        outfeat=torch.cat((outfeat,x3),1)
        outfeat=self.dec3(outfeat)
        outfeat=self.up(outfeat)
        outfeat=torch.cat((outfeat,x1),1)
        tri=self.deco(outfeat)
        outfeat=self.dec4(outfeat)
        outfeat=self.up(outfeat)
        outfeat=torch.cat((outfeat,x),1)
        outfeat=self.dec5(outfeat)
        return outfeat,tri


if __name__ == '__main__':
    a=Basicmatting().cuda()
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