import os, sys, numpy as np
import argparse
from time import time
from tqdm import tqdm
import random
import gc

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torchvision.models as models

from net import Network

parser = argparse.ArgumentParser(description='Train Virtual Screening')
parser.add_argument('train_data', metavar='DIR',
                    help='path to train dataset')
parser.add_argument('test_data', metavar='DIR',
                    help='path to test dataset')
parser.add_argument('-m', '--test', default=False, type=bool,
                    metavar='N', help='only train or add test')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=400, type=int,
                    metavar='N', help='mini-batch size (default: 400)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
args = parser.parse_args()

cos = nn.CosineSimilarity()

modelpath = './model/test/'
pdbids = open("./retrieval_proteinlist.txt", "r")

def main():
    ##  Network initialize  ##
    net = Network()  # defalt number of classes 2
    #net.load_state_dict(torch.load('./model/cs_globalmean/model_10.pth'))
    #print('Load model successfully')

    ##  define loss function (criterion) and optimizer  ##
    criterion = nn.CosineEmbeddingLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay = 1e-4)

    ##  Data loading  ##
    traindir = os.path.join(args.train_data, 'train')
    valpdir   = os.path.join(args.test_data, 'pocket')
    valldir   = os.path.join(args.test_data, 'ligand')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    num_classes = len([name for name in os.listdir(traindir)]) - 1
    print("num_classes = '{}'".format(num_classes))

    train_data = datasets.ImageFolder(  ## train/tdata, fdata
        traindir,
        transforms.Compose([
            transforms.ToTensor(),  ## (height x width, channel),(0-255) -> (channel x height x width),(0.0-1.0)
            normalize,              ## GRB の正規化
        ]))
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.workers)

    val_pdata = datasets.ImageFolder(  ## val/pocket
        valpdir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    val_ploader = torch.utils.data.DataLoader(dataset=val_pdata,
                                            batch_size=20,  # batch-size for test
                                            shuffle=False,
                                            num_workers=args.workers)

    val_ldata = datasets.ImageFolder(  ## val/ligand
        valldir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    val_lloader = torch.utils.data.DataLoader(dataset=val_ldata,
                                            batch_size=20,  # batch-size for test
                                            shuffle=False,
                                            num_workers=args.workers)

    ##  Train  ##
    print(('Start training: lr %f, batch size %d, classes %d'%(args.lr, args.batch_size, num_classes)))
    steps = args.start_epoch
    iter_per_epoch = args.batch_size//40

    ## Shuffle data
    imgs = []
    lbls = []
    image_list = []
    label_list = []
    for i, (images, labels) in enumerate(train_loader):
        imgs.append(images)
        lbls.append(labels)

    shuffle_list = [i*40 for i in range(iter_per_epoch*len(imgs))]
    random.shuffle(shuffle_list)

    list_length = iter_per_epoch*len(imgs)
    for i in range(list_length):
        s = shuffle_list[i]//args.batch_size
        f = shuffle_list[i]%args.batch_size
        image_list.append(imgs[s][f:f+40])
        label_list.append(lbls[s][f:f+40])

    init_numdict = {}
    numlist = []
    for i in range(list_length):
        if label_list[i][0].tolist()==0:
            numlist.append(i)

    for i in range(int(list_length/2)):
        init_numdict[i] = numlist[i]

    for epoch in range(args.start_epoch, args.epochs):
        if epoch%1==0 and epoch>args.start_epoch:
            path = modelpath + 'model_' + str(epoch) + '.pth'
            torch.save(net.state_dict(), path)
            print('>>>>>Save model successfully<<<<<')

        loss = 0
        sum_loss = 0

        if (epoch+1)%(int(list_length/2)-1)==0 and epoch>0:
            image_list, init_numdict = shuffle_fpair(image_list, label_list, list_length, init_numdict, 1)
            print('Shuffle mode >>>> 1')
        else:
            image_list, init_numdict = shuffle_fpair(image_list, label_list, list_length, init_numdict, 0)
            print('Shuffle mode >>>> 0')

        image_list, label_list, init_numdict = shuffle_set(image_list, label_list, list_length, init_numdict)

        for i , (images, lables) in enumerate(zip(image_list, label_list)):
            images = Variable(image_list[i])
            labels = Variable(label_list[i])

            # Forward + Backward + Optimize
            label = torch.tensor([labels[0]])
            label = label*2-1

            optimizer.zero_grad()

            output_lig, output_poc = net(images, 'train', 'max', 'max')

            sim = cos(output_lig, output_poc)
            loss += criterion(output_lig, output_poc, label.type_as(output_lig))

            if (i+1)%(iter_per_epoch)==0 and i>0:
                loss /= iter_per_epoch  ## calculate loss average
                sum_loss += loss
                print('Epoch: %2d, iter: %2d, Loss: %.4f' %(epoch, i+1, loss))
                if (i+1)==list_length:
                    print('>>>Epoch: %2d, Train_Loss: %.4f' %(epoch, sum_loss/list_length*iter_per_epoch))
                    sum_loss = 0
                    if args.test:
                        test(net, val_ploader, val_lloader, epoch)
                loss.backward()
                optimizer.step()
                loss = 0

def test(net, val_ploader, val_lloader, epoch):
    print('Evaluating network.......')
    net.eval()
    top01, top05, top10 =0, 0, 0
    for i, (pimage, plabel) in enumerate(val_ploader):
        poc_image = Variable(pimage)
        poc_label = Variable(plabel[0])
        output_poc = net(poc_image, 'pocket', 'none', 'max')
        #print('poc_label: ', poc_label)
        outputs = {}
        results = {}
        for j, (limage, llabel) in enumerate(val_lloader):
            lig_image = Variable(limage)
            lig_label = Variable(llabel[0])

            images = torch.cat((lig_image, poc_image), 0)
            ## Estimate similarity
            output_lig = net(lig_image, 'ligand', 'none', 'max')
            sim = cos(output_lig, output_poc)

            outputs[lig_label.tolist()] = sim

            ## Calculate accuracy
            if j==57:
                sortdic = sorted(outputs.items(), key=lambda x:x[1], reverse=True)
                top10_label = [l[0] for l in sortdic[0:10]]
                result = [s[0] for s in sortdic]
                #print(result)
                #print(top10_label)
                topn = find_topn(result, poc_label.tolist())
                print('Protein number %2d  finds Top %2d' %(poc_label.tolist(), topn))
                prec01, prec05, prec10 = caltop10(poc_label.tolist(), top10_label)
                top01 += prec01
                top05 += prec05
                top10 += prec10
        #print('protein: %2d, top1: %.2f%%, top5: %.2f%%, top10: %.2f%%\n' %(i, top01/(i+1)*100, top05/(i+1)*100, top10/(i+1)*100))

    print('\n\n###   Virtual Screening for %2d proteins   ###' %(i+1))
    print('Top1 Accuracy: %.4f%%, Top5 Accuracy: %.4f%%, Top10 Accuracy: %.4f%%' %(top01/(i+1)*100, top05/(i+1)*100, top10/(i+1)*100))

    net.train()

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


def shuffle_set(list1, list2, num, num_dict):
    nlist1 = []
    nlist2 =[]
    slist = [i for i in range(num)]
    fdict = {}
    random.shuffle(slist)

    for i in range(num):
        nlist1.append(list1[slist[i]])
        nlist2.append(list2[slist[i]])
        if list2[slist[i]][0].tolist()==0:
            fdict[slist[i]] = i

    for i in range(int(num/2)):
        num_dict[i] = fdict[num_dict[i]]

    return nlist1, nlist2, num_dict

## Shuffle Negative-pair
def shuffle_fpair(images, labels, num, ndict, mode):
    ligand_list = []
    new_dict = {}
    for i in range(num):
        ligand_list.append(images[i][0:20])

    if mode==0:
        for i in range(int(num/2)):
            if i==0:
                images[ndict[(int(num/2)-1)]][0:20] = ligand_list[ndict[i]]
            else:
                images[(ndict[i-1])][0:20] = ligand_list[ndict[i]]
    else:
        for i in range(int(num/2)):
            if i==0:
                images[ndict[(int(num/2)-2)]][0:20] = ligand_list[ndict[i]]
            elif i==1:
                images[ndict[(int(num/2)-1)]][0:20] = ligand_list[ndict[i]]
            else:
                images[(ndict[i-1])][0:20] = ligand_list[ndict[i]]

    for i in range(int(num/2)):
        if i==0:
            new_dict[(int(num/2)-1)] = ndict[i]
        else:
            new_dict[i-1] = ndict[i]

    return images, new_dict

if __name__ == '__main__':
    main()
