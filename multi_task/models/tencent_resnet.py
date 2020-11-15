# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch 
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x, mask):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        #out = self.linear(out)
        return out, mask


class TencentDecoder(nn.Module):
    def __init__(self, num_class=4, fc_dim=512, pool_scales=(2, 3, 4, 6), task_type='C'):
        super(TencentDecoder, self).__init__()
        # self.task_type = task_type

        # self.ppm = []
        # for scale in pool_scales:
        #     self.ppm.append(nn.Sequential(
        #         nn.AdaptiveAvgPool2d(scale),
        #         nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU(inplace=True)
        #     ))
        # self.ppm = nn.ModuleList(self.ppm)

        # self.conv_last = nn.Sequential(
        #     nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
        #               kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, num_class, kernel_size=1)
        # )
        self.fc1 = nn.Linear(10752,4096)
        self.fc2 = nn.Linear(4096,5)
        self.s = nn.Softmax()


    def forward(self, conv_out, mask):
        # conv5 = conv_out[-1]
        # conv5 = conv_out
        # input_size = conv5.size()
        # ppm_out = [conv5]
        # for pool_scale in self.ppm:
        #     ppm_out.append(nn.functional.upsample(
        #         pool_scale(conv5),
        #         (input_size[2], input_size[3]),
        #         mode='bilinear'))
        # ppm_out = torch.cat(ppm_out, 1)

        # x = self.conv_last(ppm_out)
        x = self.spatial_pyramid_pool(conv_out, 2, [int(conv_out.size(2)),int(conv_out.size(3))], [4,2,1])
        # if self.task_type == 'C':
        #     x = nn.functional.log_softmax(x, dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.s(x)
        print(x.size())
        # x = F.log_softmax(x, dim=1)
        x = x.view(-1, 5, 1)
        print(x.size())
        return x, mask


    def spatial_pyramid_pool(self,previous_conv, num_sample, previous_conv_size, out_pool_size):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer
        
        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''    
        for i in range(len(out_pool_size)):
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            h_pad = int(math.floor((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2))
            w_pad = int(math.floor((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2))
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if(i == 0):
                spp = x.view(num_sample,-1)
            else:
                spp = torch.cat((spp,x.view(num_sample,-1)), 1)
        return spp
