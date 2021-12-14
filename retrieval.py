import os, sys, numpy as np
import argparse
from time import time
from tqdm import tqdm
import gc
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torchvision.models as models
import json

import matplotlib.pyplot as plt

from attentionnet_sa import Network

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
parser.add_argument('-a', '--arch', default='alex', type=str,
                    metavar='ARCH', help='network for feature extractor')
args = parser.parse_args()

nview = 20
cos = nn.CosineSimilarity(dim=1).cuda()
pdbids = open("./retrieval_proteinlist.txt", "r")
# pdbids = open("../MN40list.txt", "r")

def main():
    ##  Network initialize  ##
    net = Network(classes=2, arch=args.arch)     ## defalt number of classes 2
    net.cuda()
    model_path = args.model
    print(model_path)
    net.load_state_dict(torch.load(model_path), strict=False)  ## load trained model
    net.eval()

    ##  Data loading  ##
    pocdir = os.path.join(args.data, 'pocket')
    ligdir = os.path.join(args.data, 'ligand')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    num_classes = len([name for name in os.listdir(pocdir)])
    # print("num_classes = '{}'".format(num_classes))kore

    pocket_data = datasets.ImageFolder(  ## pocket
        pocdir,
        transforms.Compose([
            transforms.ToTensor(),  ## (height x width, channel),(0-255) -> (channel x height x width),(0.0-1.0)
            normalize,              ## GRB の正規化
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
    # print('###   Start Screening   ###')kore
    batch_time, net_time = [], []
    sum_time = 0
    top01, top05, top10 =0, 0, 0

    with torch.no_grad():
        for i, ((pimage, plabel), pid) in enumerate(zip(pocket_loader, pdbids)):
            start_time = time()

            poc_image = Variable(pimage).cuda() # 20*3*224*224
            poc_label = Variable(plabel[0]).cuda()
            fmap_poc = net(poc_image, 0, 'pocket', args.global_pooling, args.pooling)
            # print('PDB ID: ', pid.replace('\n',''))kore
            outputs = {}
            results = {}
            for j, (limage, llabel) in enumerate(ligand_loader):
                lig_image = Variable(limage).cuda() # 20*3*224*224
                lig_label = Variable(llabel[0]).cuda()
                ## Estimate similarity
                # output_lig = net(lig_image, 'ligand', args.global_pooling, args.pooling)
                # lig_image = shuffle_imgs(lig_image)
                # poc_image = shuffle_imgs(poc_image)
                # print(lig_image.shape)
                # images = torch.cat((lig_image, poc_image), 0) # 40*3*224*224
                output_lig, output_poc, weights = net(lig_image, fmap_poc, 'attention', args.global_pooling, args.pooling)#kore
                # draw_heatmap(weights)
                # create_radarchart(weights[20:40], pid)

                # _, _, a = net(lig_image, fmap_poc, 'attention', args.global_pooling, args.pooling)
                # _, _, b = net(shuffle_imgs(lig_image), fmap_poc, 'attention', args.global_pooling, args.pooling)
                # draw_heatmap(a, "./heatmap/a")
                # draw_heatmap(b, "./heatmap/b")
                # _, _, test = net(shuffle_imgs(lig_image), fmap_poc, 'attention', args.global_pooling, args.pooling)kore
                if i==j:
                    draw_heatmap(weights, "./heatmap/model36/"+pid.replace('\n',''))
                # print(weights, '\n')
                # print(a)
                # print(b)
                # sys.exit()


                sim = cos(output_lig, output_poc)
                outputs[lig_label.tolist()] = sim

                ## Calculate accuracy
                if j==num_classes-1:
                    sortdic = sorted(outputs.items(), key=lambda x:x[1], reverse=True)
                    top10_label = [l[0] for l in sortdic[0:10]]
                    result = [s[0] for s in sortdic]
                    # print(top10_label)
                    topn = find_topn(result, poc_label.tolist())
                    print('No. %2d  ID: %s finds Top %2d' %(poc_label.tolist(), pid.replace('\n',''), topn))#kore
                    # print(top10_label)
                    prec01, prec05, prec10 = caltop10(poc_label.tolist(), top10_label)
                    top01 += prec01
                    top05 += prec05
                    top10 += prec10
            sum_time += time()-start_time
            # print('Top1: %.2f%%, Top5: %.2f%%, Top10: %.2f%%\n' %(top01/(i+1)*100, top05/(i+1)*100, top10/(i+1)*100))kore

    # print('\n\n###   Virtual Screening for %2d proteins   ###' %(i+1))kore
    # print('Top1 Accuracy: %.4f%%, Top5 Accuracy: %.4f%%, Top10 Accuracy: %.4f%%' %(top01/(i+1)*100, top05/(i+1)*100, top10/(i+1)*100))
    print('Top1 Accuracy: %.4f%%, Top5 Accuracy: %.4f%%, Top10 Accuracy: %.4f%%' %(top01/num_classes*100, top05/num_classes*100, top10/num_classes*100))
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

def shuffle_imgs(imgs):
    index = list(range(len(imgs)))
    random.shuffle(index)
    shuffled_imgs = []
    for i in index:
        shuffled_imgs.append(imgs[i])
    return torch.stack(shuffled_imgs)

def create_radarchart(weights, pid):
    view = np.array(range(1, 21))
    plt.ylim(0, 0.2)
    plt.xlabel("view_number", fontsize=16)
    plt.ylabel("Attention Weight", fontsize=16)
    plt.xticks(np.arange(1, 21, 1))
    plt.plot(view, weights)
    plt.savefig("./test3.png")
    sys.exit()

def draw_heatmap(weights, sname):
    # 描画する
    fig, ax = plt.subplots(1, 2)
    row_labels = np.array(range(1, 21))
    column_labels = np.array(range(1, 21))

    heatmap = ax[0].pcolor(weights[0:20], cmap=plt.cm.Blues)

    ax[0].set_xticks(np.arange(weights[0:20].shape[0]) + 0.5, minor=False)
    ax[0].set_yticks(np.arange(weights[0:20].shape[1]) + 0.5, minor=False)

    ax[0].invert_yaxis()
    ax[0].set_xticklabels(row_labels, minor=False, fontsize=6)
    ax[0].set_yticklabels(column_labels, minor=False, fontsize=6)
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_xlabel("ligand", fontsize=12)
    ax[0].set_ylabel("pocket", fontsize=12)

    heatmap = ax[1].pcolor(weights[20:40], cmap=plt.cm.Blues)

    ax[1].set_xticks(np.arange(weights[20:40].shape[0]) + 0.5, minor=False)
    ax[1].set_yticks(np.arange(weights[20:40].shape[1]) + 0.5, minor=False)

    ax[1].invert_yaxis()
    ax[1].set_xticklabels(row_labels, minor=False, fontsize=6)
    ax[1].set_yticklabels(column_labels, minor=False, fontsize=6)
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_xlabel("ligand", fontsize=12)
    ax[1].set_ylabel("pocket", fontsize=12)

    plt.savefig(sname+'.png')
    plt.close()

if __name__ == '__main__':
    main()
