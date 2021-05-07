import sys
import argparse
import open3d as o3d
sys.path.append('Path to python in pymol') # add path
import pymol
import pymeshlab

parser = argparse.ArgumentParser(description='Select Pocket from Pockets')
parser.add_argument('mol2dir', default='./PDB', type=str,
                    metavar='DIR', help='path to dataset')
parser.add_argument('datalist', default='./train_proteinlist.txt', type=str,
                    help='path to datalist')
parser.add_argument('savedir', default='', type=str,
                    help='path to saving directory')
parser.add_argument('--object', default='', type=str,
                    help='select object')
args = parser.parse_args()

ms = pymeshlab.MeshSet()

pdbids = open(args.datalist, "r")

if args.object=='pocket':
    footer = '_pock'
elif args.object=='ligand':
    footer = '_ligand'


for pid in pdbids:
    pymol.cmd.load(args.mol2dir + '/' + pid.replace('\n','') + footer + '.mol2')  # pdbファイルの読み込み

    pymol.cmd.hide('everything')  # hide spheres
    pymol.cmd.show('surface')  # show surface

    pymol.cmd.save(args.savedir + '/' + pid.replace('\n','') + footer + '.wrl')  # wrl形式で保存

    pymol.cmd.remove('all')
    print("create wrlfile: ", pid.replace('\n',''))

    # convert wrl-file to ply-file by pymeshlab
    ms.load_new_mesh(args.savedir + '/' + pid.replace('\n','') + footer + '.wrl')
    ms.save_current_mesh(args.savedir + '/' + pid.replace('\n','') + footer + '.ply')

pdbids.close()
