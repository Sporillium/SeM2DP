import lidar

from m2dp import createDescriptor

# Python Package imports
import numpy as np
from tqdm import trange
import argparse
import open3d as o3d
import sys

# Argument Parser:
parser = argparse.ArgumentParser(description="Process Data into M2DP Descriptors")
parser.add_argument('-n', '--number', required=True, type=int, help="Specify sequence number to be processes") # Sequence Number
parser.add_argument('-r', '--resume', type=int, help="Set a specific frame to resume execution from") # Resume frame
parser.add_argument('-s', '--use_sem', action='store_true', help="Set flag to use semantic information for descriptor generation") #Use Sem True/False
parser.add_argument('-v', '--use_velodyne', action='store_true', help="Set flag to use Velodyne data for descriptor generation") #Use Velo True/False

args = parser.parse_args()

# Define Execution flags:
SEQ_NUM = args.number
USE_SEM = args.use_sem
USE_VELO = args.use_velodyne
resume = args.resume

np.set_printoptions(threshold=sys.maxsize)

# ----- Main Execution Starts here -----
# Load processing objects:
velo_proc = lidar.VelodyneProcessor(camera_id=0, seq=SEQ_NUM)
seq_leng = velo_proc.seq_len
seq_name = f'{SEQ_NUM:02}'

if not USE_SEM and USE_VELO:
    descriptors = {}
    pcd = o3d.geometry.PointCloud()
    if resume is None:
        with open("descriptor_texts/velo_descriptors_kitti_"+seq_name+".txt", 'w') as file:
            for im in trange(seq_leng):
                print(im)
                point_cloud = velo_proc.createCloud(im)

                #pcd.points = o3d.utility.Vector3dVector(point_cloud)
                #o3d.visualization.draw_geometries([pcd])

                descriptors[im], data_rot = createDescriptor(point_cloud)
                pcd.points = o3d.utility.Vector3dVector(data_rot)
                
                #o3d.visualization.draw_geometries([pcd])

                line = np.array2string(descriptors[im], max_line_width=10000, separator=';')
                file.write(line+"\n")
    else:
        with open("descriptor_texts/velo_descriptors_kitti_"+seq_name+".txt", 'a') as file:
            for im in trange(resume, seq_leng):
                point_cloud = velo_proc.createCloud(im)
                descriptors[im] = createDescriptor(point_cloud)
                line = np.array2string(descriptors[im], max_line_width=10000, separator=';')
                file.write(line+"\n")
    
    print(len(descriptors))