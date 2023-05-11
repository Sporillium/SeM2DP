import numpy as np
import lidar
import segmentation as seg
import stereo
import argparse
from tqdm import trange

# Argument Parser:
parser = argparse.ArgumentParser(description="Process Data into M2DP Descriptors")
parser.add_argument('-n', '--number', required=True, type=int, help="Specify sequence number to be processes") # Sequence Number
parser.add_argument('-p', '--path', required=True, type=str, help="Specify path to save point cloud data") # Point Path
parser.add_argument('-s', '--sem', action='store_true', help="Store the clouds with semantic classes") # Addition of semantics

args = parser.parse_args()
# Define Execution flags:
SEQ_NUM = args.number
seq_name = f'{SEQ_NUM:02}'

save_path = args.path + seq_name + '/'

semantic_flag = args.sem
# Create Velodyne Processor
velo_proc = lidar.VelodyneProcessor(camera_id=0, seq=SEQ_NUM)

if semantic_flag:
    segmentation_engine = seg.SegmentationEngine(model_id=6, use_gpu=False)
    stereo_extractor = stereo.StereoExtractor(segmentation_engine, detector='SIFT', matcher='BF', camera_id=0, seq=SEQ_NUM)

# Load Transfromation matrices
with open(velo_proc.calib, 'r') as file:
    lines = file.readlines()
    P0 = np.fromstring(lines[2][4:], sep=' ')
    TR = np.fromstring(lines[4][4:], sep=' ')

    P0 = np.reshape(P0, (3,4))
    TR = np.reshape(TR, (3,4))
    TR = np.vstack([TR, [0, 0, 0, 1]])

# Read Velodyne Points:
if not semantic_flag:
    for im in trange(velo_proc.seq_len):
        file_name = f'{im:06}'
        #print(file_name)
        point_cloud = velo_proc.createCloudProj(im)
        # Project to image plane
        trans = P0@TR@point_cloud

        # Points in front of the camera:
        idx = (trans[2,:]>=0)
        pts2d_cam = trans[:,idx]
        pts3d_cam = point_cloud[:, idx]

        # Transform points to pixel coords
        pts2d_cam = pts2d_cam/pts2d_cam[2,:]

        pts2d_cam_x = pts2d_cam[0,:]
        pts2d_cam_y = pts2d_cam[1,:]

        # Filter Points that land on the valid image plane
        #pts2d_cam = pts2d_cam[:, (1241 >= pts2d_cam_x) & (pts2d_cam_x >= 0) & (376 >= pts2d_cam_y) & (pts2d_cam_y >= 0)]
        pts3d_cam = pts3d_cam[:, (1241 >= pts2d_cam_x) & (pts2d_cam_x >= 0) & (376 >= pts2d_cam_y) & (pts2d_cam_y >= 0)]

        pts3d_cam = pts3d_cam[:3, :].T
        pts3d_cam.astype(np.float32).tofile(save_path+file_name+'.bin')

elif semantic_flag:
    for im in trange(velo_proc.seq_len):
        file_name = f'{im:06}'
        point_cloud = velo_proc.createCloudProj(im)
        # Extract the semantic information
        sem_L, img_L = stereo_extractor.semanticsMaxFromImages(im)

        # Project to image plane
        trans = P0@TR@point_cloud

        # Points in front of the camera:
        idx = (trans[2,:]>=0)
        pts2d_cam = trans[:,idx]
        pts3d_cam = point_cloud[:, idx]

        # Transform points to pixel coords
        pts2d_cam = pts2d_cam/pts2d_cam[2,:]

        pts2d_cam_x = pts2d_cam[0,:]
        pts2d_cam_y = pts2d_cam[1,:]

        # Filter Points that land on the valid image plane
        pts2d_cam = pts2d_cam[:, (sem_L.shape[1] > pts2d_cam_x) & (pts2d_cam_x > 0) & (sem_L.shape[0] > pts2d_cam_y) & (pts2d_cam_y > 0)]
        pts3d_cam = pts3d_cam[:, (sem_L.shape[1] > pts2d_cam_x) & (pts2d_cam_x > 0) & (sem_L.shape[0] > pts2d_cam_y) & (pts2d_cam_y > 0)]

        sem_classes = sem_L[np.floor(pts2d_cam[1,:]).astype(np.int32), np.floor(pts2d_cam[0,:]).astype(np.int32)]
        colors = img_L[np.floor(pts2d_cam[1,:]).astype(np.int32), np.floor(pts2d_cam[0,:]).astype(np.int32), :]

        pts3d_cam[3, :] = sem_classes
        pts3d_cam = pts3d_cam.T
        pts3d_cam = np.hstack((pts3d_cam, colors))

        pts3d_cam.astype(np.float32).tofile(save_path+file_name+'.bin')