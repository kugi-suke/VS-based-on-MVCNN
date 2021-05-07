## 1. Download ghecom[T.kawabata, 2010]

### 1-1. Download source code

```
$ cd mkdata
```

get source code from <https://pdbj.org/ghecom/>

### 1-2. makefile

```
$ cd src

$ make
```

[T.kawabata, 2010] T.kawabata. Detection of multi-scale pockets on protein surfaces using mathematical morphology. Proteins,78, 1195-1121.

## 2. Prepare Dataset

### 2-1. Download Protein file from PDB[F.C.Bernstein et al. 1977]

```
$ cd ..

$ mkdir PDB
```

Download Protein file in ./mkdata/train_proteinlist.txt and ./mkdata/retreival_proteinlist.txt from <https://www.rcsb.org/> to ./PDB

[F.C.Bernstein et al. 1977] F.C.Bernstein, T.F.Koetzle, G.J.Williams, Jr.E.F.Meyer, M.D.Brice, J.R.Rodgers, O.Kennard, T.Shimanouchi, and M.Tasumi. The Protein Data Bank: A Computer-Based Archival File for Macromolecular Structures. JMB, Vol. 112, No. 3, pp. 535–542, 1977.

### 2-2. Pocket Detection using ghecom

```
$ mkdir Pockets

$ sh findpoc.sh -l ./train_proteinlist.txt -i ./PDB -o ./Pockets

$ sh findpoc.sh -l ./retrieval_proteinlist.txt -i ./PDB -o ./Pockets
```

### 2-3. Extract first cluster from ./Pockets to ./Pocket

```
$ mkdir Pocket

$ sh createfile.sh -l ../train_proteinlist.txt -i ./Pockets -o ./Pocket

$ sh createfile.sh -l ../retrieval_proteinlist.txt -i ./Pockets -o ./Pocket

$ python makemol2.py --object pocket ./Pocket ../train_proteinlist.txt ./Pocket
```

### 2-4. Extract Ligand from Protein file

```
$ mkdir Ligand

$ python makemol2.py --object ligand --json ../pldict.json ./PDB ../train_proteinlist.txt ./Ligand
```

### 2-5. Create TriangleMesh

```
$ mkdir Polygons

$ python createmesh.py --object pocket ./Pocket ../train_proteinlist.txt ./Polygons

$ python createmesh.py --object ligand ./Ligand ../train_proteinlist.txt ./Ligand
```

### 2-6. Create Multi-view image

Dataset for train:

```
$ python takepic.py --mode pca --object pocket --input ./Polygons --output ./train

$ python takepic.py --mode pca --object ligand --input ./Polygons --output ./train
```

Dataset for test:

```
$ python takepic.py --mode pca --object pocket --input ./Polygons --output ./test

$ python takepic.py --mode pca --object ligand --input ./Polygons --output ./test
```

## 3. Train

```
$ cd ..

$ mkdir model

$ python train.py --epochs 20 --test True ./train ./test
```

## 4. Retrieval ligand

```
$ python retrieval.py --model model_xx.pth --global_pooling none --pooling average ./test
```