import torch
import torch.nn as nn
from torch import cat, mean, max
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

import sys
sys.path.append('Utils')

cs = nn.CosineSimilarity(dim=0)

class Network(nn.Module):

    def __init__(self, classes=2, arch='alex'):
        super(Network, self).__init__()

        if arch=='res34':
            self.features_ligand = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2])
            self.features_pocket = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2])
            self.size = 512
        else:
            ## Alexnet model
            self.features_ligand = models.alexnet(pretrained=True).features
            self.features_pocket = models.alexnet(pretrained=True).features
            self.size = 256

            self.softmax = nn.Softmax(dim=0)

            self.conv1x1 = nn.Conv2d(256, 64, 1)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, y=0, mode='train', global_pooling='average', pooling='average'):
        ## Network for ligand and pocket
        if mode=='train':
            x = x.view(40, 3, 224, 224)
            lig = self.features_ligand(x[0:20])
            poc = self.features_pocket(x[20:40])

            if global_pooling=='average':
                lig = F.adaptive_avg_pool2d(lig, (1,1))
                poc = F.adaptive_avg_pool2d(poc, (1,1))
            elif global_pooling=='max':
                lig = F.adaptive_max_pool2d(lig, (1,1))
                poc = F.adaptive_max_pool2d(poc, (1,1))

            if pooling=='average':
                lig = mean(lig, dim=0)
                poc = mean(poc, dim=0)
            elif pooling=='max':
                lig = max(lig, dim=0)[0]
                poc = max(poc, dim=0)[0]

            if global_pooling=='none':
                lig = lig.view(1, self.size*6*6)
                poc = poc.view(1, self.size*6*6)
            else:
                lig = lig.view(1, self.size*1*1)
                poc = poc.view(1, self.size*1*1)
            return lig, poc, 0

        ## Network for only pocket
        elif mode=='pocket':
            poc = self.features_pocket(x)
            if global_pooling=='average':
                poc = F.adaptive_avg_pool2d(poc, (1,1))
            elif global_pooling=='max':
                poc = F.adaptive_max_pool2d(poc, (1,1))

            return poc

        ## Network for only ligand
        elif mode=='attention':
            lig = self.features_ligand(x)

            if global_pooling=='average':
                lig = F.adaptive_avg_pool2d(lig, (1,1))
            elif global_pooling=='max':
                lig = F.adaptive_max_pool2d(lig, (1,1))

            poc = y
            if pooling=='average':
                lig = mean(lig, dim=0)
                poc = mean(poc, dim=0)
            elif pooling=='max':
                lig = max(lig, dim=0)[0]
                poc = max(poc, dim=0)[0]

            if global_pooling=='none':
                lig = lig.view(1, self.size*6*6)
                poc = poc.view(1, self.size*6*6)
            else:
                lig = lig.view(1, self.size*1*1)
                poc = poc.view(1, self.size*1*1)

            return lig, poc, 0

    def get_viewWeight(self, feature):
        feature = torch.stack(feature).transpose(0, 1) # 1*40*256*1*1
        feature = self.conv1x1(feature[0]) # 40*64*1*1

        feature = feature.transpose(0, 1)# 64*40*1*1
        feature = feature.reshape(feature.size(0), -1) # 64*40

        weight = []
        for i in range(64):
            weight.append(self.fc(feature[i]))

        weight = torch.stack(weight).transpose(0,1) # 20*64
        weight = torch.sum(weight, 1) # 20*1
        weight = self.softmax(weight) # 20*1

        return weight

    def get_weightedFeature(self, feature, weight):
        wfeature = []
        for i in range(20):
            wfeature.append(torch.mul(feature[i], weight[i]))
        return torch.stack(wfeature)


def weights_init(model):
    if type(model) in [nn.Conv2d,nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)
