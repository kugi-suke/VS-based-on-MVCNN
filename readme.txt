./pdblist.txt ./PDB にあるタンパク質についてポケット検出を行い, ./Pockets に保存
sh findpoc.sh -l ./pdblist.txt -i ./PDB -o ./Pockets

./Pockets にあるタンパク質ポケット群について, 第一クラスターのみを抽出して ./Pocket に保存
sh createfile.sh -l ./pdblist.txt -i ./Pockets -o ./Pocket

./Pocket にあるタンパク質ポケットについて, ../train_proteinlist.txt を参照しながら, mol2ファイルを作成して ./Pocket に保存
python makemol2.py --object pocket ./Pocket ../train_proteinlist.txt ./Pocket

./PDB にあるタンパク質について, ../pldict.json と ../train_proteinlist.txt を参照しながら, リガンドのmol2ファイルを作成して ./Ligand に保存
python makemol2.py --object ligand --json ../pldict.json ./PDB ../train_proteinlist.txt ./Ligand

./Pocket にあるタンパク質ポケットについて, wrlファイル, plyファイルを作成して ./Mesh に保存
python createmesh.py --object pocket ./Pocket ../train_proteinlist.txt ./Mesh

./Ligand にあるリガンドについて, wrlファイル, plyファイルを作成して ./Ligand に保存
python createmesh.py --object ligand ./Ligand ../train_proteinlist.txt ./Ligand

./Mesh にあるタンパク質ポケットについて, 20方向からの写真を撮影して ./data に保存
python takepic.py
