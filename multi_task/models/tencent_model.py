import torch 
import torch.nn as nn
from torchvision import models

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
        self.patch_size = patch_size
        self.dropout = nn.Dropout(p=0.75)
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
        out = out.view(-1, 5, 1)
        return out, mask



