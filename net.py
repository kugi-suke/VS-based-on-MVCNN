# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import cat, mean, max
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

import sys
sys.path.append('Utils')
#from Layers import LRN

class Network(nn.Module):

    def __init__(self, classes=2, arch='alex'):
        super(Network, self).__init__()

        if arch=='res34':
            self.alexnet_ligand = models.resnet34(pretrained=True)
            self.alexnet_pocket = models.resnet34(pretrained=True)
        else:
            ## Alexnet model
            self.features_ligand = models.alexnet(pretrained=True).features
            self.features_pocket = models.alexnet(pretrained=True).features
            #self.alexnet_ligand = models.alexnet(pretrained=True)
            #self.alexnet_pocket = models.alexnet(pretrained=True)

        #self.apply(weights_init)

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
                    z = self.features_ligand(x[i])
                    z = F.adaptive_avg_pool2d(z, (1,1))
                    l_list.append(z)
                for i in range(20, 40):
                    z = self.features_pocket(x[i])
                    z = F.adaptive_avg_pool2d(z, (1,1))
                    p_list.append(z)

            elif global_pooling=='max':
                for i in range(20):
                    z = self.features_ligand(x[i])
                    z = F.adaptive_max_pool2d(z, (1,1))
                    l_list.append(z)
                for i in range(20, 40):
                    z = self.features_pocket(x[i])
                    z = F.adaptive_max_pool2d(z, (1,1))
                    p_list.append(z)

            elif global_pooling=='none':
                for i in range(20):
                    z = self.features_ligand(x[i])
                    l_list.append(z)
                for i in range(20, 40):
                    z = self.features_pocket(x[i])
                    p_list.append(z)

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
                    z = self.features_pocket(x[i])
                    z = F.adaptive_avg_pool2d(z, (1,1))
                    p_list.append(z)

            elif global_pooling=='max':
                for i in range(20):
                    z = self.features_pocket(x[i])
                    z = F.adaptive_max_pool2d(z, (1,1))
                    p_list.append(z)

            elif global_pooling=='none':
                for i in range(20):
                    z = self.features_pocket(x[i])
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
                    z = self.features_ligand(x[i])
                    z = F.adaptive_avg_pool2d(z, (1,1))
                    l_list.append(z)

            elif global_pooling=='max':
                for i in range(20):
                    z = self.features_ligand(x[i])
                    z = F.adaptive_max_pool2d(z, (1,1))
                    l_list.append(z)

            elif global_pooling=='none':
                for i in range(20):
                    z = self.features_ligand(x[i])
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
