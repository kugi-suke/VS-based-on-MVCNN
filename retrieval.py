import os, sys, numpy as np
import argparse
from time import time
from tqdm import tqdm
import gc

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torchvision.models as models
import json

from net import Network

parser = argparse.ArgumentParser(description='Train Virtual Screening')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 20)')
parser.add_argument('-p', '--pooling', default='average', type=str,
                    metavar='N', help='type of pooling')
parser.add_argument('-g', '--global_pooling', default='average', type=str,
                    metavar='N', help='type of global pooling')
parser.add_argument('-m', '--model', default='', type=str,
                    metavar='N', help='using trained model')
args = parser.parse_args()

nview = 20
cos = nn.CosineSimilarity()
pdbids = open("./retrieval_proteinlist.txt", "r")

def main():
    ##  Network initialize  ##
    net = Network()                              ## defalt number of classes 2
    model_path = args.model
    print(model_path)
    net.load_state_dict(torch.load(model_path))  ## load trained model
    print('Load model Successfully')
    net.eval()

    ##  Data loading  ##
    pocdir = os.path.join(args.data, 'pocket')
    ligdir = os.path.join(args.data, 'ligand')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    num_classes = len([name for name in os.listdir(pocdir)]) - 1
    print("num_classes = '{}'".format(num_classes))

    pocket_data = datasets.ImageFolder(  ## pocket
        pocdir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    pocket_loader = torch.utils.data.DataLoader(dataset=pocket_data,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.workers)

    ligand_data = datasets.ImageFolder(  ## ligand
        ligdir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    ligand_loader = torch.utils.data.DataLoader(dataset=ligand_data,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.workers)

    ##  Virtual Screening  ##
    print('###   Start Screening   ###')
    batch_time, net_time = [], []
    sum_time = 0
    top01, top05, top10 =0, 0, 0
    for i, ((pimage, plabel), pid) in enumerate(zip(pocket_loader, pdbids)):
        start_time = time()

        poc_image = Variable(pimage)
        poc_label = Variable(plabel[0])
        output_poc = net(poc_image, 'pocket', args.global_pooling, args.pooling)
        print('PDB ID: ', pid.replace('\n',''))
        outputs = {}
        results = {}
        for j, (limage, llabel) in enumerate(ligand_loader):
            lig_image = Variable(limage)
            lig_label = Variable(llabel[0])

            ## Estimate similarity
            output_lig = net(lig_image, 'ligand', args.global_pooling, args.pooling)
            sim = cos(output_lig, output_poc)
            outputs[lig_label.tolist()] = sim

            ## Calculate accuracy
            if j==57:
                sortdic = sorted(outputs.items(), key=lambda x:x[1], reverse=True)
                top10_label = [l[0] for l in sortdic[0:10]]
                result = [s[0] for s in sortdic]
                #print(result)
                print(top10_label)
                topn = find_topn(result, poc_label.tolist())
                print('No. %2d  finds Top %2d' %(poc_label.tolist(), topn))
                prec01, prec05, prec10 = caltop10(poc_label.tolist(), top10_label)
                top01 += prec01
                top05 += prec05
                top10 += prec10
        sum_time += time()-start_time
        print('Top1: %.2f%%, Top5: %.2f%%, Top10: %.2f%%\n' %(top01/(i+1)*100, top05/(i+1)*100, top10/(i+1)*100))

    print('\n\n###   Virtual Screening for %2d proteins   ###' %(i+1))
    print('Top1 Accuracy: %.4f%%, Top5 Accuracy: %.4f%%, Top10 Accuracy: %.4f%%' %(top01/(i+1)*100, top05/(i+1)*100, top10/(i+1)*100))
    print(sum_time)
    pdbids.close()

def caltop10(correct_label, top10_label):
    if correct_label==top10_label[0]:
        return 1, 1, 1
    elif top10_label[1:5].count(correct_label)==1:
        return 0, 1, 1
    elif top10_label[5:10].count(correct_label)==1:
        return 0, 0, 1
    else:
        return 0, 0, 0


def find_topn(result, label):

    for i, output in enumerate(result):
        if output==label:
            return i+1
    return -1  ## error

if __name__ == '__main__':
    main()
