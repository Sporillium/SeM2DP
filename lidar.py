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
                self.poses = camera_info['poses_path']+f'{seq:02}'+'.txt'
                self.seq_len = len[seq]
            else:
                self.poses = camera_info['poses_path']
                self.seq_len = camera_info['seq_len']    
            f.close()
    # Methods
    def createCloud(self, im_no):
        image_str = f'{im_no:06}'
        point_cloud = []
        cloud = np.fromfile(self.path_velo+image_str+'.bin', dtype=np.float32).reshape((4,-1))
        #print(cloud.shape[0], cloud.shape[1])
        for i in range(cloud.shape[1]):
            row = cloud[:,i]
            #print(row.shape)
            #print(row)
            point_loc = row[:3]
            #print(point_loc.shape)
            point_ref = row[3]
            point = Point(point_loc, point_ref)
            point_cloud.append(point)
        
        return point_cloud


class Point:
    def __init__(self, location, reflect):
        self.location = location
        self.reflect = reflect