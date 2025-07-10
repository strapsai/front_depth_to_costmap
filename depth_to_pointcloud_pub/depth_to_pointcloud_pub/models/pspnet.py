#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
logging.getLogger('torch').setLevel(logging.ERROR)
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import math
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
from torchvision import models
from einops import rearrange
from timm.models.vision_transformer import _cfg

import sys
sys.path.insert(0, "/home/park/Documents/ws_yspark/toss/RADIO")

from hubconf import radio_model

def fix_seed(seed=1):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # current GPU
    torch.cuda.manual_seed_all(seed)  # all GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(1)  # 원하는 seed 값


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)
    
class pspnet_encoder_ssl(nn.Module):

    def __init__(
            self,
            layers=50,
            bins=(1, 2, 3, 6),
            dropout=0.1,
            classes=256,
            zoom_factor=8,
            use_ppm=True,
            criterion=nn.CrossEntropyLoss(ignore_index=255),
            pretrained=True,
            final_dim=768
        ):
        super(pspnet_encoder_ssl, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        resnet = resnet50(pretrained=True)
        
        # 🔧 기존 conv1 저장
        old_conv = resnet.conv1  # [64, 3, 7, 7]

        # 🔁 layer0에 새 conv1을 적용
        self.layer0 = nn.Sequential(
            old_conv,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            # fea_dim *= 2

        self.local_proj = nn.Conv2d(2048, final_dim, kernel_size=1, bias=False)  # local: 2048 → 1024

        self.cls = nn.Sequential(
            nn.Conv2d(final_dim*2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1, bias=False)
        )

    def forward(self, x, y, detach=False):
        x_size = x.size()
        assert (x_size[2]) % 8 == 0 and (x_size[3]) % 8 == 0

        h = int((x_size[2]) / 8 * self.zoom_factor)
        w = int((x_size[3]) / 8 * self.zoom_factor)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)		
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        x = self.ppm(x)
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        local_feat = x[:, 2048:, :, :]  # local part
        local_feat = self.local_proj(local_feat)  # → (B, 1024, 32, 32)

        x = torch.cat([y, local_feat], dim=1)
        x_out = self.cls(x)

        return (x_out, x) #x_out

def inject_pretrained_cfg(model):
    cfg = _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/ViT-H_16.npz',
        input_size=(3, 512, 512),
        interpolation='bicubic',
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        num_classes=1000,
        crop_pct=1.0,
        first_conv='patch_embed.proj',
        classifier='head',
    )
    model.default_cfg = cfg
    model.pretrained_cfg = cfg
    return model

class radio_encoder_ssl(nn.Module):
    def __init__(
        self,
        model_version='radio_v2.5-b',  # ViT-H/16 기반 RADIO 2.5
        pretrained=True,
        dropout=0.1,
        classes=2,
        zoom_factor=8,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        return_layer=7,  # 7번째 block (index=6) 통과한 feature를 사용
        ppm_bins=(1, 2, 3, 6),
        ppm_reduction_dim=320,
        final_dim=768,
    ):
        super(radio_encoder_ssl, self).__init__()
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.return_layer = return_layer

        # RADIO 모델 로드 (torch hub)
        self.backbone = radio_model(
            version=model_version,
            pretrained=pretrained,
            skip_validation=True
        )

        version_to_dim = {
            'radio_v2.1': 432,
            'radio_v2.5-b': 768,
            'radio_v2.5-l': 1024,
            'radio_v2.5-h': 1280,
            'e-radio': 512
        }

        # 2. pretrained_cfg 수동 등록
        self.backbone.model = inject_pretrained_cfg(self.backbone.model)

        # 출력 feature dimension (radio_v2.5-h 기준 1280)
        self.fea_dim = version_to_dim[model_version]

        self.patch_embed = nn.Conv2d(
            in_channels=3,  # input은 3채널
            out_channels=self.fea_dim,  # output
            kernel_size=16,  # ViT-H/16이면 patch size가 16
            stride=16
        )

        # self.global_proj = nn.Conv2d(self.fea_dim, final_dim, kernel_size=1, bias=False)

    def forward(self, x, detach=False):
        x_size = x.size()
        h = int(x_size[2] / 8 * self.zoom_factor)
        w = int(x_size[3] / 8 * self.zoom_factor)

        # 1. input_conditioner 통과
        x = self.backbone.input_conditioner(x)

        # 2. patch embedding 직접 수행 (Conv2d 사용)
        x = self.patch_embed(x)  # (B, 1280, H/16, W/16)

        # 3. flatten + transpose
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # 4. blocks 0~7번 통과
        for i in range(8):
            x = self.backbone.model.blocks[i](x)

        # 5. norm
        x = self.backbone.model.norm(x)

        # 6. (B, N, C) → (B, C, H, W) 변환
        B, N, C = x.shape           # N = H*W  (H, W 는 patch_embed 단계에서 얻음)

        x = (
            x.permute(0, 2, 1)      # (B, C, N)
            .contiguous()         # ★ ONNX 필수: 메모리 연속화
            .view(B, C, H, W)     # or .reshape(B, C, H, W)
        )
        # x = self.global_proj(x)  # → (B, 1024, 32, 32)

        return x

        # x = self.ppm(x)  # (B, 2560, 14, 14)

        # # 7. Classification head
        # out = self.cls(x.detach() if detach else x)

        # if self.zoom_factor != 1:
        #     out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        # return (out, x)



# [Rank 0] feature_extractor.backbone.model.blocks.7.norm1.weight shape: torch.Size([1280])                                                                     
# [Rank 0] feature_extractor.backbone.model.blocks.7.norm1.bias shape: torch.Size([1280])                                                                       
# [Rank 0] feature_extractor.backbone.model.blocks.7.attn.qkv.weight shape: torch.Size([3840, 1280])                                                            
# [Rank 0] feature_extractor.backbone.model.blocks.7.attn.qkv.bias shape: torch.Size([3840])                                                                    
# [Rank 0] feature_extractor.backbone.model.blocks.7.attn.proj.weight shape: torch.Size([1280, 1280])                                                           
# [Rank 0] feature_extractor.backbone.model.blocks.7.attn.proj.bias shape: torch.Size([1280])                                                                   
# [Rank 0] feature_extractor.backbone.model.blocks.7.norm2.weight shape: torch.Size([1280])                                                                     
# [Rank 0] feature_extractor.backbone.model.blocks.7.norm2.bias shape: torch.Size([1280])                                                                       
# [Rank 0] feature_extractor.backbone.model.blocks.7.mlp.fc1.weight shape: torch.Size([5120, 1280])                                                             
# [Rank 0] feature_extractor.backbone.model.blocks.7.mlp.fc1.bias shape: torch.Size([5120])                                                                     
# [Rank 0] feature_extractor.backbone.model.blocks.7.mlp.fc2.weight shape: torch.Size([1280, 5120])                                                             
# [Rank 0] feature_extractor.backbone.model.blocks.7.mlp.fc2.bias shape: torch.Size([1280])  
