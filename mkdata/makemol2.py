import sys
import argparse
sys.path.append('/home/kugimoto/.local/pymol-open-source-build/lib/python') # pymolをインポートするためにpath追加
import pymol
import json

parser = argparse.ArgumentParser(description='Select Pocket from Pockets')
parser.add_argument('pdbdir', default='./PDB', type=str,
                    metavar='DIR', help='path to dataset')
parser.add_argument('datalist', default='./train_proteinlist.txt', type=str,
                    help='path to datalist')
parser.add_argument('savedir', default='', type=str,
                    help='path to saving directory')
parser.add_argument('--json', default='', type=str,
                    help='path to protein-ligand json')
parser.add_argument('--object', default='', type=str,
                    help='select object')
args = parser.parse_args()

pdbids = open(args.datalist, "r")  # read protein-list

##  Pocket(Protein) case  ##
if args.object=='pocket':
    for pid in pdbids:
        pymol.cmd.load(args.pdbdir + '/' + pid.replace('\n','') + '_pock.pdb')  # pdbファイルの読み込み
        pymol.cmd.save(args.savedir + '/' + pid.replace('\n','') + '_pock.mol2')  # wrl形式で保存
        pymol.cmd.remove('all')
        print("create mol2file: ", pid.replace('\n',''))

##  Ligand case  ##
elif args.object == 'ligand':
    json_open = open(args.json, 'r')
    ligand = json.load(json_open)
    for pid in pdbids:
        pymol.cmd.load(args.pdbdir + '/' + pid.replace('\n','') + '.pdb')  # pdbファイルの読み込み
        pymol.cmd.remove("resn hoh")
        command = 'resn ' + ligand[pid.replace('\n','')]
        pymol.cmd.select("ligand", command)
        pymol.cmd.save(args.savedir + '/' + pid.replace('\n','') + '_ligand.mol2', "ligand")  # wrl形式で保存

pdbids.close()
