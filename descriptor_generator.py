# Code for evaluating different loop closure methods against ground truth

# Custom File Imports:
import segmentation as seg
import stereo
import cloud
import lidar

from m2dp import createDescriptor
from sem2dp import createSemDescriptor, des_compress_new

# Python Package imports
import numpy as np
from tqdm import trange
import argparse

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

# ----- Main Execution Starts here -----
# Load processing objects:

segmentation_engine = seg.SegmentationEngine(model_id=6, use_gpu=False)
stereo_extractor = stereo.StereoExtractor(segmentation_engine, detector='SIFT', matcher='BF', camera_id=0, seq=SEQ_NUM)
cloud_engine = cloud.CloudProcessor(stereo_extractor)
velo_proc = lidar.VelodyneProcessor(camera_id=0, seq=SEQ_NUM)
seq_leng = stereo_extractor.seq_len
seq_name = f'{SEQ_NUM:02}'

#pool = multiprocessing.Pool(6)

if not USE_SEM and not USE_VELO:
    descriptors = {}
    with open("descriptor_texts/basic_descriptors_kitti_"+seq_name+".txt", 'w') as file:
        for im in trange(seq_leng):
            point_cloud = cloud_engine.processFrameNoSemantics(im)
            cloud_arr = []
            for point in point_cloud:
                loc = point.location
                cloud_arr.append(loc)
            cloud_arr = np.asarray(cloud_arr)
            descriptors[im] = createDescriptor(cloud_arr)
            line = np.array2string(descriptors[im], max_line_width=10000, separator=';')
            file.write(line+"\n")

    print(len(descriptors))

if USE_SEM and not USE_VELO:
    descriptors = {}
    if resume is None:
        with open("descriptor_texts/sem_descriptors_kitti_"+seq_name+".txt", 'w') as file:
            for im in trange(seq_leng):
                point_cloud, labels = cloud_engine.processFrame(im)
                descriptors[im] = createDescriptor(point_cloud)
                sem_descriptor = createSemDescriptor(point_cloud, labels)
                comp_sem_descriptor, descriptor_size = des_compress_new(sem_descriptor)
                line1 = np.array2string(descriptors[im], max_line_width=10000, separator=';')
                line2 = np.array2string(comp_sem_descriptor, max_line_width=50000, separator=';', threshold=10000)
                file.write(line1+"\n"+line2+"\n")
    else:
        with open("descriptor_texts/sem_descriptors_kitti_"+seq_name+".txt", 'a') as file:
            for im in trange(resume, seq_leng):
                point_cloud, labels = cloud_engine.processFrame(im)
                descriptors[im] = createDescriptor(point_cloud)
                sem_descriptor = createSemDescriptor(point_cloud, labels)
                comp_sem_descriptor, descriptor_size = des_compress_new(sem_descriptor)
                line1 = np.array2string(descriptors[im], max_line_width=10000, separator=';')
                line2 = np.array2string(comp_sem_descriptor, max_line_width=50000, separator=';', threshold=10000)
                file.write(line1+"\n"+line2+"\n")

    print(len(descriptors))
    print(descriptor_size)

if not USE_SEM and USE_VELO:
    descriptors = {}
    if resume is None:
        with open("descriptor_texts/velo_descriptors_kitti_"+seq_name+".txt", 'w') as file:
            for im in trange(seq_leng):
                point_cloud = velo_proc.createCloud(im)
                descriptors[im] = createDescriptor(point_cloud)
                line = np.array2string(descriptors[im], max_line_width=10000, separator=';')
                file.write(line+"\n")
    else:
        with open("descriptor_texts/velo_descriptors_kitti_"+seq_name+".txt", 'a') as file:
            for im in trange(resume, seq_leng):
                point_cloud = velo_proc.createCloud(im)
                descriptors[im] = createDescriptor(point_cloud)
                line = np.array2string(descriptors[im], max_line_width=10000, separator=';')
                file.write(line+"\n")

if USE_SEM and USE_VELO:
    descriptors = {}
    if resume is None:
        with open("descriptor_texts/mod_velo_descriptors_kitti_"+seq_name+".txt", 'w') as file:
            for im in trange(seq_leng):
                point_cloud = velo_proc.createCloudMod(im)
                descriptors[im] = createDescriptor(point_cloud)
                line = np.array2string(descriptors[im], max_line_width=10000, separator=';')
                file.write(line+"\n")
    else:
        with open("descriptor_texts/mod_velo_descriptors_kitti_"+seq_name+".txt", 'a') as file:
            for im in trange(resume, seq_leng):
                point_cloud = velo_proc.createCloudMod(im)
                descriptors[im] = createDescriptor(point_cloud)
                line = np.array2string(descriptors[im], max_line_width=10000, separator=';')
                file.write(line+"\n")
    
    print(len(descriptors))

