import math

import torch
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn as nn
# models.
# from models.attention import BasicBlock, SEBasicBlock, ECALayerBlock, LayerNorm, Scale_Fusion, Scale_Fusion2

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, expansion=2, downsample=None):
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes * self.expansion, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        # if sigmoid:
        #     self.sigmoid = nn.Sigmoid()
        # else:
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ECALayerBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(ECALayerBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, inplanes, stride)
        self.group_conv3x3 = nn.Conv1d(inplanes, inplanes, kernel_size=7, stride=1,
                                       padding=7 // 2, groups=inplanes // 16, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(inplanes)
        self.se = SELayer(inplanes, reduction)
        self.ECA = ECALayer(inplanes)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.group_conv3x3(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.ECA(out)

        out = residual + out
        return out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, inplanes, stride)
        self.group_conv3x3 = nn.Conv1d(inplanes, inplanes, kernel_size=3, stride=1,
                                       padding=1, groups=inplanes // 16, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(inplanes, reduction)

    def forward(self, x):
        residual = x
        out = self.group_conv3x3(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.se(out)
        out = residual + out
        return out


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class Scale_Fusion(nn.Module):
    def __init__(self, x1_channels, x2_channels, ):
        super(Scale_Fusion, self).__init__()

        self.up = nn.Sequential(
            nn.Conv1d(x1_channels, x2_channels, kernel_size=3, stride=2, bias=False, padding=1),
            nn.BatchNorm1d(x2_channels),
            nn.ReLU(),
        )

        self.down = nn.Sequential(
            nn.Conv1d(x2_channels, x1_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x1_channels),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
        )

        self.layer1 = nn.Sequential(
            BasicBlock(x1_channels, x1_channels, expansion=1),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            BasicBlock(x1_channels, x1_channels, expansion=1),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            BasicBlock(x1_channels, x1_channels, expansion=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(x2_channels, x2_channels, expansion=1),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            BasicBlock(x2_channels, x2_channels, expansion=1),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            BasicBlock(x2_channels, x2_channels, expansion=1),
        )

    def forward(self, x1, x2):
        up1 = self.up(x1)
        down1 = self.down(x2)
        feat1 = down1 + x1
        feat2 = up1 + x2
        x1 = self.layer1(feat1)
        x2 = self.layer2(feat2)
        return x1, x2

class Scale_split(nn.Module):
    def __init__(self, x1_channels, x2_channels, ):
        super(Scale_split, self).__init__()

        self.up = nn.Sequential(
            nn.Conv1d(x1_channels, x2_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x2_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.down = nn.Sequential(
            nn.Conv1d(x2_channels, x1_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x1_channels),
            nn.Upsample(size=127, mode='nearest'),
        )

        self.layer1 = nn.Sequential(
            BasicBlock(x1_channels, x1_channels, expansion=1),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            BasicBlock(x1_channels, x1_channels, expansion=1),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            BasicBlock(x1_channels, x1_channels, expansion=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(x2_channels, x2_channels, expansion=1),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            BasicBlock(x2_channels, x2_channels, expansion=1),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            BasicBlock(x2_channels, x2_channels, expansion=1),
        )

    def forward(self, x1, x2):
        up1 = self.up(x1)
        down1 = self.down(x2)
        feat1 = down1 + x1
        feat2 = up1 + x2
        x1 = self.layer1(feat1)
        x2 = self.layer2(feat2)
        return x1, x2


class Scale_Fusion2(nn.Module):
    def __init__(self, x1_channels, x2_channels, x3_channels):
        super(Scale_Fusion2, self).__init__()

        self.up1_2 = nn.Sequential(
            nn.Conv1d(x1_channels, x2_channels, kernel_size=3, stride=2, bias=False, padding=1),
            nn.BatchNorm1d(x2_channels),
            nn.ReLU(),
        )

        self.up1_3 = nn.Sequential(
            nn.Conv1d(x1_channels, x3_channels, kernel_size=3, stride=2, bias=False, padding=1),
            nn.BatchNorm1d(x3_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.up2_3 = nn.Sequential(
            nn.Conv1d(x2_channels, x3_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x3_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.down2_1 = nn.Sequential(
            nn.Conv1d(x2_channels, x1_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x1_channels),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
        )

        self.down3_1 = nn.Sequential(
            nn.Conv1d(x3_channels, x1_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x1_channels),
            nn.Upsample(scale_factor=4.0, mode='nearest'),
        )

        self.down3_2 = nn.Sequential(
            nn.Conv1d(x3_channels, x2_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x2_channels),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
        )

        self.layer1 = nn.Sequential(
            BasicBlock(x1_channels, x1_channels, expansion=1),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            BasicBlock(x1_channels, x1_channels, expansion=1),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            BasicBlock(x1_channels, x1_channels, expansion=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(x2_channels, x2_channels, expansion=1),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            BasicBlock(x2_channels, x2_channels, expansion=1),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            BasicBlock(x2_channels, x2_channels, expansion=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(x3_channels, x3_channels, expansion=1),
            SEBasicBlock(x3_channels),
            ECALayerBlock(x3_channels),
            BasicBlock(x3_channels, x3_channels, expansion=1),
            SEBasicBlock(x3_channels),
            ECALayerBlock(x3_channels),
            BasicBlock(x3_channels, x3_channels, expansion=1),
        )

    def forward(self, x1, x2, x3):
        # up1_2 = self.up1_2(x1)
        # up1_3 = self.up1_3(x1)
        # up2_3 = self.up2_3(x2)
        # down2_1 = self.down2_1(x2)
        # down3_1 = self.down3_1(x3)
        # down3_2 = self.down3_2(x3)
        # x1 = down2_1 + down3_1 + x1
        # x2 = up1_2 + down3_2 + x2
        # x3 = up1_3 + up2_3 + x3
        # x1 = self.layer1(x1)
        # x2 = self.layer2(x2)
        # x3 = self.layer3(x3)

        up1_2 = self.up1_2(x1)
        up1_3 = self.up1_3(x1)
        up2_3 = self.up2_3(x2)
        # down2_1 = self.down2_1(x2)
        # down3_1 = self.down3_1(x3)
        # down3_2 = self.down3_2(x3)
        x1 = x1
        x2 = up1_2 + x2
        x3 = up1_3 + up2_3 + x3
        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        x3 = self.layer3(x3)

        return x1, x2, x3

class BaseScale(nn.Module):
    def __init__(self, input_channels, mid_channels, stride=1):
        super(BaseScale, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, mid_channels, kernel_size=3, stride=stride, bias=False, padding=3 // 2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        filter_sizes = [3, 13, 23, 33]
        # filter_sizes = [23,23,23,23]
        self.conv1 = nn.Conv1d(input_channels, mid_channels, kernel_size=filter_sizes[0],
                               stride=stride, bias=False, padding=(filter_sizes[0] // 2))
        self.conv2 = nn.Conv1d(input_channels, mid_channels, kernel_size=filter_sizes[1],
                               stride=stride, bias=False, padding=(filter_sizes[1] // 2))
        self.conv3 = nn.Conv1d(input_channels, mid_channels, kernel_size=filter_sizes[2],
                               stride=stride, bias=False, padding=(filter_sizes[2] // 2))
        self.conv4 = nn.Conv1d(input_channels, mid_channels, kernel_size=filter_sizes[3],
                               stride=stride, bias=False, padding=(filter_sizes[3] // 2))
        self.bn = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU()
        self.dro = nn.Dropout(0.2)
        self.max = nn.MaxPool1d(2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x_concat = torch.mean(torch.stack([x1, x2, x3, x4], 2), 2)
        x_concat = self.dro(self.max(self.relu(self.bn(x_concat))))
        x5 = self.conv_block(x)
        x = x5 + x_concat
        return x


class BaseBlock(nn.Module):
    def __init__(self, input_channels=12, mid_channels=16, stride=1):
        super(BaseBlock, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, mid_channels, kernel_size=11, stride=1, bias=False, padding=11 // 2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels * 2, kernel_size=7, stride=1, bias=False, padding=7 // 2),
            nn.BatchNorm1d(mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(mid_channels * 2, mid_channels * 4, kernel_size=5, stride=1, bias=False, padding=5 // 2),
            nn.BatchNorm1d(mid_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(mid_channels * 4, mid_channels * 4, kernel_size=3, stride=1, bias=False, padding=3 // 2),
            nn.BatchNorm1d(mid_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv1d(mid_channels * 4, mid_channels * 4, kernel_size=3, stride=1, bias=False, padding=3 // 2),
            nn.BatchNorm1d(mid_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.BaseScale1 = BaseScale(mid_channels * 4, mid_channels * 4, stride=1)
        self.BaseScale2 = BaseScale(mid_channels * 4, mid_channels * 8, stride=2)
        self.layer1 = nn.Sequential(
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            # ECALayerBlock(mid_channels * 4),
            # SEBasicBlock(mid_channels * 4),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            # ECALayerBlock(mid_channels * 4),
            # SEBasicBlock(mid_channels * 4),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x1 = self.BaseScale1(x)
        x2 = self.BaseScale2(x)
        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        return x1, x2


class Fusion_block(nn.Module):
    def __init__(self, input_channels, kernel_size=7, c=1):
        super(Fusion_block, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.norm1 = LayerNorm(input_channels * 2, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(input_channels * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(input_channels * 3, eps=1e-6, data_format="channels_first")
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels * 2, input_channels, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(input_channels),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(1),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(input_channels, input_channels // 8, 1, bias=False),
            nn.Conv1d(input_channels // 8, input_channels, 1, bias=False),
            nn.BatchNorm1d(input_channels),
        )
        self.conv1 = nn.Conv1d(input_channels, input_channels, kernel_size=1, )
        self.gelu = nn.GELU()
        self.fc = nn.Linear(372 // c, 250 // c)

        self.residual = BasicBlock(input_channels * 3, input_channels)
        self.residual2 = BasicBlock(input_channels * 2, input_channels)
        self.drop_path = nn.Identity()
        downsample = nn.Sequential(
            nn.Conv1d(input_channels * 2, input_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(input_channels)
        )
        self.layer1 = nn.Sequential(
            BasicBlock(input_channels * 2, input_channels, expansion=1, downsample=downsample),
            ECALayerBlock(input_channels),
            SEBasicBlock(input_channels),
            BasicBlock(input_channels, input_channels, expansion=1),
        )

    def forward(self, l):
        max_result = self.maxpool(l)
        avg_result = self.avgpool(l)
        max_out = self.conv_block3(max_result)
        avg_out = self.conv_block3(avg_result)
        l1 = self.sigmoid(max_out + avg_out) * l

        max_result, _ = torch.max(l, dim=1, keepdim=True)
        avg_result = torch.mean(l, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        l2 = self.conv_block2(result)
        l2 = self.sigmoid(l2) * l

        fuse = torch.cat([l1, l2], 1)
        fuse = self.norm1(fuse)
        fuse = self.layer1(fuse)
        fuse = self.drop_path(fuse)
        return fuse


class Dui1(nn.Module):

    def __init__(self, input_channels=12, mid_channels=16, mu2=1, sigma2=0.1, num_classes=5):
        super(Dui1, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Fusion_block1 = Fusion_block(64, c=1)
        self.Fusion_block2 = Fusion_block(128, c=1)
        self.conv_norm1 = nn.LayerNorm(64, eps=1e-6)
        self.conv_norm2 = nn.LayerNorm(128, eps=1e-6)
        self.BaseBlock = BaseBlock(input_channels=input_channels)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(64, num_classes)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x1, x2 = self.BaseBlock(x)
        x1 = self.Fusion_block1(x1)
        x2 = self.Fusion_block2(x2)
        feat1 = x1.view(x1.size(0), -1)
        feat2 = x2.view(x2.size(0), -1)
        x1 = self.avg_pool(x1).squeeze()
        x2 = self.avg_pool(x2).squeeze()
        out1 = self.fc1(self.conv_norm1(x1))
        out2 = self.fc2(self.conv_norm2(x2))
        return feat1, out1, feat2, out2

class CNN(nn.Module):

    def __init__(self, input_channels=12, mid_channels=16, mu2=1, sigma2=0.1, num_classes=5, branch=2):
        super(CNN, self).__init__()
        self.branch = branch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Fusion_block1 = Fusion_block(64, c=1)
        self.Fusion_block2 = Fusion_block(128, c=1)
        self.Fusion_block3 = Fusion_block(512, c=1)
        self.conv_norm1 = nn.LayerNorm(64, eps=1e-6)
        self.conv_norm2 = nn.LayerNorm(128, eps=1e-6)
        self.conv_norm3 = nn.LayerNorm(128 * 4, eps=1e-6)
        self.BaseBlock = BaseBlock(input_channels=input_channels)
        self.layer2 = nn.Sequential(
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
        )
        self.layer1 = nn.Sequential(
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            ECALayerBlock(mid_channels * 4),
            SEBasicBlock(mid_channels * 4),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            ECALayerBlock(mid_channels * 4),
            SEBasicBlock(mid_channels * 4),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.down1 = nn.Conv1d(128, 128, kernel_size=1, stride=1, bias=False)
        self.down2 = nn.Conv1d(256, 128, kernel_size=1, stride=1, bias=False)
        self.scale_f = Scale_Fusion(128, 256)
        self.scale_f2 = Scale_Fusion2(128, 256, 512)
        self.up1_2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=2, bias=False, padding=3 // 2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(64, num_classes)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc3 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1, x2 = self.BaseBlock(x)
        x2 = self.Fusion_block2(x2)
        feat2 = x2.view(x2.size(0), -1)
        x2 = self.avg_pool(x2).squeeze()
        out = self.fc2(self.conv_norm2(x2))
        return out

# class CNN(nn.Module):

#     def __init__(self, input_channels=12, mid_channels=16, mu2=1, sigma2=0.1, num_classes=5, branch=2):
#         super(CNN, self).__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.attentionblock= Fusion_block(128, c=1)
#         self.LN = nn.LayerNorm(128, eps=1e-6)
#         self.MultiscaleConv = BaseBlock(input_channels=input_channels)
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(128, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         _, x = self.MultiscaleConv(x)
#         x = self.attentionblock(x)
#         feat = x.view(x.size(0), -1)
#         x = self.avg_pool(x).squeeze()
#         out = self.fc(self.LN(x))
#         return out

def MRMnet(**kwargs):
    model = Dui1(input_channels=12, mid_channels=16, mu2=1, sigma2=0.1, **kwargs)
    return model
