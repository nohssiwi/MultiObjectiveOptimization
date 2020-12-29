import torch 
import torch.nn as nn
from torchvision import models

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
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
        else:
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp

class TencentEncoder(nn.Module) :
    def __init__(self) :
        super(TencentEncoder, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((4, 4))
    
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
        x = x.view(x.size(0), -1)
        return x, mask

class TencentDecoder(nn.Module):
    def __init__(self, patch_size):
        super(TencentDecoder, self).__init__()
        self.patch_size = patch_size + 1
        self.dropout = nn.Dropout(p=0.75)
        self.fc = nn.Linear(8192, 5)# resnet18
        # self.fc = nn.Linear(32768, 5)# resnet50
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
        if (self.patch_size > 0) :
            out = self.aggragate(out)
        out = out.view(-1, 5, 1)
        return out, mask



