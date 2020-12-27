# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import math

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def spatial_pyramid_pool(self,previous_conv, num_sample, previous_conv_size, out_pool_size):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer
        
        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''    
        for i in range(len(out_pool_size)):
            h_wid = math.ceil(previous_conv_size[0] / out_pool_size[i])
            w_wid = math.ceil(previous_conv_size[1] / out_pool_size[i])
            h_pad = math.floor((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2)
            w_pad = math.floor((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2)
            # torch.nn.functional.pad
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if(i == 0):
                spp = x.view(num_sample,-1)
            else:
                spp = torch.cat((spp,x.view(num_sample,-1)), 1)
        return spp

    def forward(self, x, mask):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        if(self.patch_size > 0) :
            out = out.view(out.size(0), -1)
        else :
            out = out.view(out.size(0), -1)
            # out = self.spatial_pyramid_pool(out, out.size(0), [int(out.size(2)), int(out.size(3))], [3,2,1])
        return out, mask


class TencentEncoder(nn.Module) :
    def __init__(self) :
        super(TencentEncoder, self).__init__()
        self.model = models.resnet18(pretrained=True)

    def spatial_pyramid_pool(self,previous_conv, num_sample, previous_conv_size, out_pool_size):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer
        
        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''    
        for i in range(len(out_pool_size)):
            h_wid = math.ceil(previous_conv_size[0] / out_pool_size[i])
            w_wid = math.ceil(previous_conv_size[1] / out_pool_size[i])
            h_pad = math.floor((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2)
            w_pad = math.floor((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2)
            # torch.nn.functional.pad
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if(i == 0):
                spp = x.view(num_sample,-1)
            else:
                spp = torch.cat((spp,x.view(num_sample,-1)), 1)
        return spp
    
    def forward(self, x, mask):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        return x, mask

class TencentDecoder(nn.Module):
    def __init__(self, patch_size):
        super(TencentDecoder, self).__init__()
        self.patch_size = patch_size
        self.dropout = nn.Dropout(p=0.75)
        if(self.patch_size > 0) :
            self.fc = nn.Linear(10752, 5)
        else :
            self.fc = nn.Linear(8192, 5)
        self.s = nn.Softmax(dim=1)

    def aggragate(self, patches) :
        out = patches.reshape(-1, self.patch_size, 5)
        out = torch.sum(out, dim=1) / self.patch_size
        return out


    def forward(self, conv_out, mask):
        out = self.dropout(conv_out)
        # print(out.shape)
        out = self.fc(out)
        out = self.s(out)
        if(self.patch_size > 0) :
            out = self.aggragate(out)
        out = out.view(-1, 5, 1)
        return out, mask



