import sys
import subprocess
import argparse
import open3d as o3d
import numpy as np
import math as m
import os
import random

import sklearn
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='Take picture for 3Dmodel')
parser.add_argument('-i', '--input', default='./Polygons', type=str,
                    metavar='DIR', help='path to input directory')
parser.add_argument('-o', '--output', default='./train/', type=str,
                    metavar='DIR', help='path output directory')
parser.add_argument('--object', default='', type=str,
                    metavar='N', help='select object')
parser.add_argument('-m', '--mode', default='', type=str,
                    metavar='N', help='select mode default, random or pca')
args = parser.parse_args()

phi = (1+m.sqrt(5))/2
phi2 = phi**2

## vertex point of Dodecahedron
vertex=[
[0, -1, -phi2],      # 0
[0,  1, -phi2],      # 1
[0, -1,  phi2],      # 2
[0,  1,  phi2],      # 3
[-1, -phi2, 0],      # 4
[ 1, -phi2, 0],      # 5
[-1,  phi2, 0],      # 6
[ 1,  phi2, 0],      # 7
[-phi2, 0, -1],      # 8
[-phi2, 0,  1],      # 9
[ phi2, 0, -1],      # 10
[ phi2, 0,  1],      # 11
[-phi, -phi, -phi],  # 12
[-phi, -phi,  phi],  # 13
[-phi,  phi, -phi],  # 14
[-phi,  phi,  phi],  # 15
[ phi, -phi, -phi],  # 16
[ phi, -phi,  phi],  # 17
[ phi,  phi, -phi],  # 18
[ phi,  phi,  phi]   # 19
]

vertex = np.array(vertex)

def main():
    idir = args.idir
    odir = args.odir

    if args.object=='pocket':  ## Pocket-file's footter and header
        rfile_footer = '_pock.ply'
        wfile_header = 'pocket_'

    elif args.object=='ligand':  ## Ligand-file's footter and header
        rfile_footer = '_ligand.ply'
        wfile_header = 'ligand_'

    else:
        print('Select object: pocket or ligand')  ## error
        sys.exit()

    pdbids = open("./retrieval_proteinlist.txt", "r")

    ## Take Pictures
    for id in pdbids:
        giom_pc=o3d.io.read_point_cloud(idir + id.replace('\n','') + rfile_footer)  ## read ply file
        giom_tm=o3d.io.read_triangle_mesh(idir + id.replace('\n','') + rfile_footer)
        o3d.geometry.TriangleMesh.paint_uniform_color(giom_tm,[0.5,0.5,0.5])
        giom2 = o3d.geometry.TriangleMesh.compute_triangle_normals(o3d.geometry.TriangleMesh.compute_vertex_normals(giom_tm))

        ## initialize pose(pca or random or default)
        R = init_pose(args.mode, giom_pc)
        ct = o3d.geometry.TriangleMesh.get_center(giom_tm)
        giom2.rotate(R, center=ct)

        dirpath = odir + id.replace('\n','')
        os.makedirs(dirpath, exist_ok=True)
        ## save images
        for j in range(20):
            save_img(giom2, dirpath+'/', wfile_header + id.replace('\n','')+'_001_0'+str(j+1), vertex[j])

    pdbids.close()

def save_img(source, file_path, file_name, view_point):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=224, height=224)
    ctr = vis.get_view_control()
    vis.add_geometry(source)
    ctr.set_front(view_point)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(file_path + "/" + file_name + ".png")
    vis.destroy_window()
    del ctr
    del vis

def init_pose(mode, giom_pc):
    if mode=='pca':
        g_array = np.asarray(giom_pc.points)
        pca = PCA(n_components=3, whiten=False)
        pca.fit(g_array)
        p_arr = np.concatenate([[pca.components_[0], pca.components_[1], pca.components_[2]]],0)
        inv_p = np.linalg.inv(p_arr)

        return inv_p.T

    elif mode=='random':
        randx = random.uniform(0, 360) * m.pi/180
        randy = random.uniform(0, 360) * m.pi/180
        randz = random.uniform(0, 360) * m.pi/180
        R = o3d.geometry.get_rotation_matrix_from_xyz((randx, randy, randz))

        return R

    else:
        return np.identity(3)

if __name__ == '__main__':
    main()
