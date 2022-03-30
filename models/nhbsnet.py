# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.resnet import resnet50
from models.basenet import CAM_Module, PAM_Module, FeatureFusion

num_classes = 8
bn_eps = 1e-5
bn_momentum = 0.1


def get():
    return NHBSNet(num_classes, None, None)


class NHBSNet(nn.Module):
    def __init__(self, n_classes, n_channels, cuda_device=True, pretrained_model=None, norm_layer=nn.BatchNorm2d):
        super(NHBSNet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.device = cuda_device
        self.context_path = resnet50(pretrained_model, norm_layer=norm_layer,
                                     bn_eps=bn_eps,
                                     bn_momentum=bn_momentum,
                                     deep_stem=False, stem_width=64)

        self.refine3x3 = ConvBnRelu(2048, 512, 3, 1, 1,
                                    has_bn=True, norm_layer=norm_layer,
                                    has_relu=True, has_bias=False)
        self.refine1x1 = ConvBnRelu(512, 512, 3, 1, 1,
                                    has_bn=True, norm_layer=norm_layer,
                                    has_relu=True, has_bias=False)
        #self.fpa = FPA(channels=512)
        self.CA = CAM_Module(in_dim=512)
        self.PA = PAM_Module(in_dim=512)
        # self.final_FFM = FeatureFusion(in_planes=1024, out_planes=512)
        self.FFM = FeatureFusion(in_planes=1024, out_planes=512)
        self.low_FFM = FeatureFusion(in_planes=1024, out_planes=1024)
        self.output_head = OutputHead(1026, n_classes, 8, True, norm_layer)

    def forward(self, data):
        # spatial_out = self.spatial_path(data)

        context_blocks = self.context_path(data)
        context_blocks.reverse()

        refine = self.refine3x3(context_blocks[0])
        refine = self.refine1x1(refine)
        # FPA = self.fpa(refine)
        # final_fm = self.final_FFM(refine, FPA)
        ca = self.CA(refine)
        pa = self.PA(refine)
        ffm = self.FFM(ca, pa)
        fm = F.interpolate(ffm, size=context_blocks[2].shape[2:], mode="bilinear", align_corners=False)
        # h = torch.cat((fm, context_blocks[2]), dim=1)
        h = self.low_FFM(fm, context_blocks[2])
        x_range = torch.linspace(-1, 1, h.shape[-1])
        y_range = torch.linspace(-1, 1, h.shape[-2])
        Y, X = torch.meshgrid(y_range, x_range)
        Y = Y.expand([h.shape[0], 1, -1, -1])
        X = X.expand([h.shape[0], 1, -1, -1])
        if self.device == True:
            coord_feat = torch.cat([X, Y], 1).cuda()
            h = torch.cat([h, coord_feat], 1).cuda()
        else:
            coord_feat = torch.cat([X, Y], 1)
            h = torch.cat([h, coord_feat], 1)
        h = self.output_head(h)
        return h


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class OutputHead(nn.Module):
    def __init__(self, in_planes, n_classes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(OutputHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 256, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(256, n_classes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, n_classes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)

        return output


if __name__ == "__main__":
    model = NHBSNet(n_classes=8, n_channels=3, cuda_device=False)
    image = torch.randn(1, 3, 256, 256)
    label = torch.randn(1, 8, 256, 256)
    pred = model(image)
    print(pred.size())
