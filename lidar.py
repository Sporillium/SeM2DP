# File for handling Velodyne LIDAR Point Clouds

# Imports
import numpy as np
import yaml
from yaml.loader import SafeLoader

# Class definition
class VelodyneProcessor:
    # Constructor
    def __init__(self, camera_id=0, seq=None):
        self.camera_id = camera_id
        self.seq = seq

        with open('./model_info/camera_info.yaml') as f:
            cameras = yaml.load(f, Loader=SafeLoader)['ID']
            camera_info = cameras[camera_id]
            self.cam_name = camera_info['name']
            
            if self.cam_name == 'kitti': # Check for Kitti Dataset and load correct sequence info
                sequences = camera_info['seq_id']
                len = camera_info['seq_len']
                self.path_velo = camera_info['image_path']+sequences[seq]+'velodyne/'
                self.path_velo_mod = camera_info['image_path']+sequences[seq]+'modified_velodyne/'
                self.path_velo_sem = camera_info['image_path']+sequences[seq]+'semantic_velodyne/'
                self.poses = camera_info['poses_path']+f'{seq:02}'+'.txt'
                self.seq_len = len[seq]
                self.calib = camera_info['calib_path']+sequences[seq]+'calib.txt'
            else:
                self.poses = camera_info['poses_path']
                self.seq_len = camera_info['seq_len']    
            f.close()
    # Methods
    def createCloud(self, im_no):
        image_str = f'{im_no:06}'
        cloud = np.fromfile(self.path_velo+image_str+'.bin', dtype=np.float32).reshape((-1, 4))
        point_cloud = cloud[:, :3]
        #print(point_cloud.shape)
        return point_cloud
    
    def createCloudProj(self, im_no):
        image_str = f'{im_no:06}'
        cloud = np.fromfile(self.path_velo+image_str+'.bin', dtype=np.float32).reshape((-1, 4))
        cloud[:,3] = 1
        #print(point_cloud.shape)
        return cloud.T

    def createCloudMod(self, im_no):
        image_str = f'{im_no:06}'
        cloud = np.fromfile(self.path_velo_mod+image_str+'.bin', dtype=np.float32).reshape((-1, 3))
        return cloud

    def createCloudSem(self, im_no):
        image_str = f'{im_no:06}'
        cloud = np.fromfile(self.path_velo_sem+image_str+'_sem.bin', dtype=np.float32).reshape((-1, 4))
        return cloud

class Point:
    def __init__(self, location, reflect):
        self.location = location
        self.reflect = reflect