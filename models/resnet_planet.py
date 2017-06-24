import torch
import torch.nn as nn
from torchvision import models


class PlanetNet(nn.Module):
            def __init__(self,pretrain_net_name = 'resnet50'):
                super(PlanetNet, self).__init__()
                self.pretrain_net = eval('models.'+pretrain_net_name+'(pretrained=True)')
                '''
                for param in self.pretrain_net.parameters():
                	param.requires_grad = False
                '''
                self.pretrain_net.fc = nn.Linear(2048,17)
            def forward(self, x):
                x = self.pretrain_net(x)
                return x
