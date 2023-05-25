import lidar

# Python Package imports
import numpy as np
from tqdm import trange
import argparse
import open3d as o3d
import sys

from scipy.io import loadmat
from mit_semseg.utils import colorEncode
colour_path = '/home/march/devel/DNNs/semantic-segmentation-pytorch/data/color150.mat'

colours = loadmat(colour_path)['colors']

# Argument Parser:
parser = argparse.ArgumentParser(description="Process Data into M2DP Descriptors")
parser.add_argument('-n', '--number', required=True, type=int, help="Specify sequence number to be processes") # Sequence Number
parser.add_argument('-s', '--use_sem', action='store_true', help="Set flag to use semantic information for descriptor generation") #Use Sem True/False

args = parser.parse_args()

# Define Execution flags:
SEQ_NUM = args.number
USE_SEM = args.use_sem

np.set_printoptions(threshold=sys.maxsize)

# ----- Main Execution Starts here -----
# Load processing objects:
velo_proc = lidar.VelodyneProcessor(camera_id=0, seq=SEQ_NUM)
seq_leng = velo_proc.seq_len
seq_name = f'{SEQ_NUM:02}'

pcd = o3d.geometry.PointCloud()
if USE_SEM:
    point_cloud = velo_proc.createCloudSem(0)
    classes = point_cloud[:,3].reshape((point_cloud.shape[0], 1))
    pred_color = np.squeeze(colorEncode(classes, colours).astype(np.uint8))
    pred_color = pred_color/255

    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pred_color)
    o3d.visualization.draw_geometries([pcd])
    
else:
    point_cloud = velo_proc.createCloud(0)
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    o3d.visualization.draw_geometries([pcd])