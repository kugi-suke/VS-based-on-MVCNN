import os, sys, numpy as np
from time import time
import random
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import Dataset

import mlflow
import hydra
from omegaconf import DictConfig

from net import Network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cos = nn.CosineSimilarity().to(device)

modelpath = 'path_to_save_model_directory'

class MVIDataset(Dataset):
    def __init__(self, mvi_dir, transform=None):
        self.dir_path = mvi_dir
        self.img_paths = [str(p) for p in Path(self.dir_path).glob("**/*.png")]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)//40

    def __getitem__(self, index):
        imgs_lig = []
        imgs_poc = []
        paths_lig =[]
        paths_poc = []
        for i in range(20):
            path = self.img_paths[index*40 + i]

            img = Image.open(path)
            if self.transform is not None:
                img = self.transform(img)
            imgs_lig.append(img)
            paths_lig.append(path.split('/')[-1])

        for i in range(20, 40):
            path = self.img_paths[index*40 + i]

            img = Image.open(path)
            if self.transform is not None:
                img = self.transform(img)
            imgs_poc.append(img)
            paths_poc.append(path.split('/')[-1])

        pdbid = path.split('/')[-3]

        return imgs_lig, imgs_poc, paths_lig, paths_poc, pdbid

@hydra.main(config_path='../configs', config_name='config.yaml')
def main(cfg):
    config = cfg
    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment('mlflow-test')

    ##  Network initialize  ##
    net = Network(classes=2, arch=cfg.NETWORK.ARCH)  # defalt number of classes 2
    net.to(device)
    if cfg.TRAIN.CHECKPOINT_FILE_PATH!='':
        net.load_state_dict(torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH))
        print('Load model successfully')

    ##  define loss function (criterion) and optimizer  ##
    criterion = nn.CosineEmbeddingLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.TRAIN.LEARNING_RATE, momentum=0.9, weight_decay = 1e-4)

    ##  Data loading  ##
    traindir = os.path.join(cfg.TRAIN.DATASET)
    valpdir   = os.path.join(cfg.VAL.DATASET, 'pocket')
    valldir   = os.path.join(cfg.VAL.DATASET, 'ligand')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                    transforms.ToTensor(),  ## (height x width, channel),(0-255) -> (channel x height x width),(0.0-1.0)
                    normalize,              ## GRB の正規化
                ])
    train_classes = len([name for name in os.listdir(traindir)])# - 1
    val_classes = len([name for name in os.listdir(valpdir)])# - 2
    print("train_classes = '{}', val_classes = '{}'".format(train_classes, val_classes))

    train_data = MVIDataset(traindir, transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=cfg.DATA_LOADER.NUM_WORKERS)

    val_pdata = datasets.ImageFolder(  ## val/pocket
        valpdir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    val_ploader = torch.utils.data.DataLoader(dataset=val_pdata,
                                            batch_size=20,  # batch-size for test
                                            shuffle=False,
                                            num_workers=cfg.DATA_LOADER.NUM_WORKERS)

    val_ldata = datasets.ImageFolder(  ## val/ligand
        valldir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    val_lloader = torch.utils.data.DataLoader(dataset=val_ldata,
                                            batch_size=20,  # batch-size for test
                                            shuffle=False,
                                            num_workers=cfg.DATA_LOADER.NUM_WORKERS)

    ##  Train  ##
    print(('Start training: lr %f, batch size %d, classes %d, with %s'%(cfg.TRAIN.LEARNING_RATE, cfg.TRAIN.BATCH_SIZE, train_classes, str(device))))

    imgs_poc = []
    imgs_lig = []
    lbls = []
    for i, (images1, images2, paths1, paths2, labels) in enumerate(train_loader):
        imgs_lig.append(images1)
        imgs_poc.append(images2)
        lbls.append(labels[0])

    batch_set_size = cfg.TRAIN.BATCH_SIZE//40
    best_acc = [0, 0, 0]
    active_run = mlflow.active_run()
    # mlflow.log_params(vars(args))
    metrics = {'train_loss': 0.0, 'val_acc_top01': 0.0, 'val_acc_top05': 0.0, 'val_acc_top10': 0.0}
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.START_EPOCH+cfg.TRAIN.EPOCHS):

        loss = 0
        sum_loss = 0
        index = get_train_index(train_classes)
        for i in range(train_classes*2):
            label_lig = index[i][0]
            label_poc = index[i][1]

            images = torch.cat([torch.stack(imgs_lig[label_lig], dim=0), torch.stack(imgs_poc[label_poc], dim=0)], dim=0)
            images = images.to(device)

            optimizer.zero_grad()
            output_lig, output_poc, w = net(images, 0, 'train', cfg.NETWORK.GP_TRAIN, cfg.NETWORK.P_TRAIN)

            loss += criterion(output_lig, output_poc, torch.tensor(index[i][2]).to(device))

            if (i+1)%(batch_set_size)==0 and i>0:
                loss /= batch_set_size  ## calculate loss average
                sum_loss += loss

                if (i+1)==train_classes*2:
                    metrics['val_acc_top01'], metrics['val_acc_top05'], metrics['val_acc_top10'] = test(cfg.NETWORK, net, val_classes, val_ploader, val_lloader)
                    metrics['train_loss'] = float(sum_loss/batch_set_size)
                    mlflow.log_metrics(metrics)

                    print('>> Epoch: %3d, ' %(epoch+1), metrics)
                    sum_loss = 0

                    if is_improved(best_acc, val_acc):
                        best_acc = np.max([best_acc, val_acc], axis=0).tolist()
                        path = modelpath + 'model_' + str(epoch) + '.pth'
                        torch.save(net.state_dict(), path)
                        print('   *')
                    else:
                        print('')

                loss.backward()
                optimizer.step()
                loss = 0

def test(cfgnet, net, val_classes, val_ploader, val_lloader):
    pdbids = open("/home/kugimoto/VS/AMVCNN-SE/retrieval_proteinlist.txt", "r")
    net.eval()
    top01, top05, top10 = 0, 0, 0
    sum_time = 0

    with torch.no_grad():
        for i, ((pimage, plabel), pid) in enumerate(zip(val_ploader, pdbids)):
            start_time = time()

            poc_image = Variable(pimage).cuda() # 20*3*224*224
            poc_label = Variable(plabel[0]).cuda()
            fmap_poc = net(poc_image, 0, 'pocket', cfgnet.GP_VAL, cfgnet.P_VAL)

            outputs = {}
            results = {}
            for j, (limage, llabel) in enumerate(val_lloader):
                lig_image = Variable(limage).cuda() # 20*3*224*224
                lig_label = Variable(llabel[0]).cuda()

                output_lig, output_poc, weights = net(lig_image, fmap_poc, 'attention', cfgnet.GP_VAL, cfgnet.P_VAL)

                sim = cos(output_lig, output_poc)
                outputs[lig_label.tolist()] = sim

                ## Calculate accuracy
                if j==val_classes-1:
                    sortdic = sorted(outputs.items(), key=lambda x:x[1], reverse=True)
                    top10_label = [l[0] for l in sortdic[0:10]]
                    prec01, prec05, prec10 = caltop10(poc_label.tolist(), top10_label)
                    top01 += prec01
                    top05 += prec05
                    top10 += prec10
            sum_time += time()-start_time

    net.train()
    pdbids.close()
    return top01/val_classes, top05/val_classes, top10/val_classes

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

def get_train_index(length):
    index = list(range(length))*2
    status = [-1 for _ in range(length)] + [1 for _ in range(length)]

    tmplist = [i for i in range(length*2)]
    random.shuffle(tmplist)
    new_index = []
    for i in range(length*2):
        if status[tmplist[i]]==1:
            new_index.append([index[tmplist[i]], index[tmplist[i]], 1])
        else:
            lot_list = list(range(0, index[tmplist[i]])) + list(range(index[tmplist[i]]+1, length))
            new_index.append([index[tmplist[i]], random.choice(lot_list), -1])

    return new_index

def is_improved(best, current):
    return best[0]<current[0] or best[1]<current[1] or best[2]<current[2]

if __name__ == '__main__':
    main()
