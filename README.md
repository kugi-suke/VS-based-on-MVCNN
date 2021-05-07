**1. Download ghecom[T.kawabata, 2010]**

1-1. Download source code

get source code from <https://pdbj.org/ghecom/>

1-2. makefile

`$ cd src`

`$ make`

[T.kawabata, 2010] T.kawabata. Detection of multi-scale pockets on protein surfaces using mathematical morphology. Proteins,78, 1195-1121.

**2. Prepare Dataset**

2-1. Download Protein file from PDB[F.C.Bernstein et al. 1977]

`$ mkdir PDB`

Download Protein file in ./mkdata/train_proteinlist.txt and ./mkdata/retreival_proteinlist.txt from <https://www.rcsb.org/> to ./PDB

[F.C.Bernstein et al. 1977] F.C.Bernstein, T.F.Koetzle, G.J.Williams, Jr.E.F.Meyer, M.D.Brice, J.R.Rodgers, O.Kennard, T.Shimanouchi, and M.Tasumi. The Protein Data Bank: A Computer-Based Archival File for Macromolecular Structures. JMB, Vol. 112, No. 3, pp. 535–542, 1977.

2-2. Pocket Detection using ghecom

`mkdir Pockets`

`sh ./mkdata/findpoc.sh -l ./train_proteinlist.txt -i ./PDB -o ./Pockets`

`sh ./mkdata/findpoc.sh -l ./retrieval_proteinlist.txt -i ./PDB -o ./Pockets`