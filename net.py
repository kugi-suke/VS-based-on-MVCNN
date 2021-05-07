# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:57:01 2017

@author: Biagio Brattoli
"""
import torch
import torch.nn as nn
from torch import cat, mean, max
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

import sys
sys.path.append('Utils')

class Network(nn.Module):

    def __init__(self, classes=2):
        super(Network, self).__init__()

        ## Alexnet model
        self.alexnet_ligand = models.alexnet(pretrained=True)
        self.alexnet_pocket = models.alexnet(pretrained=True)

    def forward(self, x, mode, global_pooling, pooling):
        x = x.numpy()
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x)
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)

        ## Network for ligand and pocket
        if mode=='train':
            p_list = []
            l_list = []

            if global_pooling=='average':
                for i in range(20):
                    z = self.alexnet_ligand.features(x[i])  ## feature extractor
                    z = F.adaptive_avg_pool2d(z, (1,1))     ## global pooling layer
                    l_list.append(z)
                for i in range(20, 40):
                    z = self.alexnet_pocket.features(x[i])
                    z = F.adaptive_avg_pool2d(z, (1,1))
                    p_list.append(z)

            elif global_pooling=='max':
                for i in range(20):
                    z = self.alexnet_ligand.features(x[i])
                    z = F.adaptive_max_pool2d(z, (1,1))
                    l_list.append(z)
                for i in range(20, 40):
                    z = self.alexnet_pocket.features(x[i])
                    z = F.adaptive_max_pool2d(z, (1,1))
                    p_list.append(z)

            elif global_pooling=='none':
                for i in range(20):
                    z = self.alexnet_ligand.features(x[i])
                    l_list.append(z)
                for i in range(20, 40):
                    z = self.alexnet_pocket.features(x[i])
                    p_list.append(z)

            ## View Pooling layer
            if pooling=='average':
                lig = mean(torch.stack(l_list), dim=0)
                poc = mean(torch.stack(p_list), dim=0)
            elif pooling=='max':
                lig = max(torch.stack(l_list), dim=0)[0]
                poc = max(torch.stack(p_list), dim=0)[0]

            if global_pooling=='none':
                lig = lig.view(lig.size(0), 256*6*6)
                poc = poc.view(poc.size(0), 256*6*6)
            else:
                lig = lig.view(lig.size(0), 256*1*1)
                poc = poc.view(poc.size(0), 256*1*1)

            return lig, poc

        ## Network for only pocket
        elif mode=='pocket':
            p_list = []

            if global_pooling=='average':
                for i in range(20):
                    z = self.alexnet_pocket.features(x[i])
                    z = F.adaptive_avg_pool2d(z, (1,1))
                    p_list.append(z)

            elif global_pooling=='max':
                for i in range(20):
                    z = self.alexnet_pocket.features(x[i])
                    z = F.adaptive_max_pool2d(z, (1,1))
                    p_list.append(z)

            elif global_pooling=='none':
                for i in range(20):
                    z = self.alexnet_pocket.features(x[i])
                    p_list.append(z)

            if pooling=='average':
                poc = mean(torch.stack(p_list), dim=0)
            elif pooling=='max':
                poc = max(torch.stack(p_list), dim=0)[0]

            if global_pooling=='none':
                poc = poc.view(poc.size(0), 256*6*6)
            else:
                poc = poc.view(poc.size(0), 256*1*1)

            return poc

        ## Network for only ligand
        elif mode=='ligand':
            l_list = []

            if global_pooling=='average':
                for i in range(20):
                    z = self.alexnet_ligand.features(x[i])
                    z = F.adaptive_avg_pool2d(z, (1,1))
                    l_list.append(z)

            elif global_pooling=='max':
                for i in range(20):
                    z = self.alexnet_ligand.features(x[i])
                    z = F.adaptive_max_pool2d(z, (1,1))
                    l_list.append(z)

            elif global_pooling=='none':
                for i in range(20):
                    z = self.alexnet_ligand.features(x[i])
                    l_list.append(z)

            if pooling=='average':
                lig = mean(torch.stack(l_list), dim=0)
            elif pooling=='max':
                lig = max(torch.stack(l_list), dim=0)[0]

            if global_pooling=='none':
                lig = lig.view(lig.size(0), 256*6*6)
            else:
                lig = lig.view(lig.size(0), 256*1*1)

            return lig

def weights_init(model):
    if type(model) in [nn.Conv2d,nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)
