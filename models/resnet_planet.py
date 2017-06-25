import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PlanetNet(nn.Module):
    def __init__(self,pretrain_net_name = 'resnet50'):
        super(PlanetNet, self).__init__()
        self.pretrain_net = eval('models.'+pretrain_net_name+'(pretrained=True)')
        self.basenet = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.dropout1 = nn.Dropout(p = 0.5)
        '''
        for param in self.pretrain_net.parameters():
        	param.requires_grad = False
        '''
        self.fc = nn.Linear(2048,17)
    def forward(self, x):
        y = self.basenet(x)
        y = self.dropout1(y)
        y = self.fc(y.view(y.size(0),-1))
        return y