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
            self.features_ligand = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2])
            self.features_pocket = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2])
            self.size = 512
        else:
            ## Alexnet model
            self.features_ligand = models.alexnet(pretrained=True).features
            self.features_pocket = models.alexnet(pretrained=True).features
            # self.classifier_ligand = nn.Sequential(*list(models.alexnet(pretrained=True).classifier.children())[:-1])
            # self.classifier_pocket = nn.Sequential(*list(models.alexnet(pretrained=True).classifier.children())[:-1])
            self.fc = nn.Linear(in_features=1, out_features=1, bias=True)
            self.softmax = nn.Softmax(dim=0)
            self.size = 256

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
            lw_list = []
            pw_list = []
            if global_pooling=='average':
                for i in range(20):
                    z = self.features_ligand(x[i])
                    z = F.adaptive_avg_pool2d(z, (1,1))
                    l_list.append(z) # 20*(1*256*1*1)
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
                wlig = self.get_viewWeight(l_list) # 20*1*1*1
                wpoc = self.get_viewWeight(p_list)
                l_list = torch.stack(l_list) # 20*1*256*1*1
                p_list = torch.stack(p_list) # 20*1*256*1*1

                lw_list = self.get_weightedFeature(l_list, wlig)
                pw_list = self.get_weightedFeature(p_list, wpoc)

                lig = mean(torch.stack(lw_list), dim=0)
                poc = mean(torch.stack(pw_list), dim=0)
            elif pooling=='max':
                lig = max(torch.stack(l_list), dim=0)[0]
                poc = max(torch.stack(p_list), dim=0)[0]

            if global_pooling=='none':
                lig = lig.view(lig.size(0), self.size*6*6)
                poc = poc.view(poc.size(0), self.size*6*6)
            else:
                lig = lig.view(lig.size(0), self.size*1*1)
                poc = poc.view(poc.size(0), self.size*1*1)

            return lig, poc

        ## Network for only pocket
        elif mode=='pocket':
            p_list = []
            pw_list = []

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
                    zw = F.adaptive_avg_pool2d(z, (1,1))
                    p_list.append(z)
                    pw_list.append(zw)

            wpoc = self.get_viewWeight(pw_list)
            p_list = torch.stack(p_list) # 20*1*256*1*1
            pw_list = self.get_weightedFeature(p_list, wpoc)
            if pooling=='average':
                wpoc = mean(torch.stack(pw_list), dim=0)
            elif pooling=='max':
                wpoc = max(torch.stack(pw_list), dim=0)[0]

            if global_pooling=='none':
                #print(poc.shape)
                wpoc = wpoc.view(wpoc.size(0), self.size*6*6)
            else:
                wpoc = wpoc.view(wpoc.size(0), self.size*1*1)

            return wpoc

        ## Network for only ligand
        elif mode=='ligand':
            l_list = []
            lw_list = []

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
                    zw = F.adaptive_avg_pool2d(z, (1,1))
                    l_list.append(z)
                    lw_list.append(zw)

            wlig = self.get_viewWeight(lw_list)
            l_list = torch.stack(l_list) # 20*1*256*1*1
            lw_list = self.get_weightedFeature(l_list, wlig)
            if pooling=='average':
                wlig = mean(torch.stack(lw_list), dim=0)
            elif pooling=='max':
                wlig = max(torch.stack(lw_list), dim=0)[0]

            if global_pooling=='none':
                wlig = wlig.view(wlig.size(0), self.size*6*6)
            else:
                wlig = wlig.view(wlig.size(0), self.size*1*1)

            return wlig

    def get_viewWeight(self, feature):
        feature = torch.stack(feature).transpose(0,2) # 256*(1*20*1*1)
        weight = []
        for i in range(256):
            weight.append(self.fc(feature[i]))

        weight = torch.stack(weight).transpose(0,2) # 20*(1*256*1*1)
        weight = torch.sum(weight, 2) # 20*1*1*1
        weight = self.softmax(weight) # 20*1*1*1

        return weight

    def get_weightedFeature(self, feature, weight):
        wfeature = []
        for i in range(20):
            wfeature.append(torch.mul(feature[i], weight[i]))
        return wfeature


def weights_init(model):
    if type(model) in [nn.Conv2d,nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)
