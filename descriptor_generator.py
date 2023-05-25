# Code for evaluating different loop closure methods against ground truth

# Custom File Imports:
import segmentation as seg
import stereo
import cloud
import lidar

from m2dp import createDescriptor, createColorDescriptor
from sem2dp import createSemDescriptor, des_compress_new, createSemDescriptorHisto

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
parser.add_argument('-m', '--use_modified', action='store_true', help="Set flag to use Modified Velodyne data for descriptor generation") # Use Mod Velo True/False
parser.add_argument('-c', '--use_color', action='store_true', help="Set flag to use color data for c-M2DP descriptor") # Use Color true / false

args = parser.parse_args()

# Define Execution flags:
SEQ_NUM = args.number
USE_SEM = args.use_sem
USE_VELO = args.use_velodyne
USE_MOD = args.use_modified
USE_COL = args.use_color
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

if not USE_SEM and not USE_VELO and not USE_COL and not USE_MOD: # Using Visual Point Clouds Only
    print("CREATING DESCRIPTORS WITH VISUAL POINTS ONLY")
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

if USE_SEM and not USE_VELO and not USE_COL and not USE_MOD: #Using Visual Semantic Point Clouds
    print("CREATING DESCRIPTORS WITH VISUAL-SEMANTIC POINTS, ARGMAX METHOD")
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

if USE_SEM and not USE_VELO and not USE_COL and USE_MOD: #Using Visual Semantic Point Clouds (Histo Mode)
    print("CREATING DESCRIPTORS WITH VISUAL-SEMANTIC POINTS, HISTOGRAM METHOD")
    descriptors = {}
    if resume is None:
        with open("descriptor_texts/sem_histo_vis_descriptors_kitti_"+seq_name+".txt", 'w') as file:
            for im in trange(seq_leng):
                point_cloud, labels = cloud_engine.processFrame(im)
                descriptors[im] = createSemDescriptorHisto(point_cloud, labels)
                line1 = np.array2string(descriptors[im], max_line_width=50000, separator=';', threshold=10000)
                file.write(line1+"\n")
    else:
        with open("descriptor_texts/sem_histo_vis_descriptors_kitti_"+seq_name+".txt", 'a') as file:
            for im in trange(resume, seq_leng):
                point_cloud, labels = cloud_engine.processFrame(im)
                descriptors[im] = createSemDescriptorHisto(point_cloud, labels)
                line1 = np.array2string(descriptors[im], max_line_width=50000, separator=';', threshold=10000)
                file.write(line1+"\n")
    print(len(descriptors))

if not USE_SEM and USE_VELO and not USE_MOD and not USE_COL: # Use Velodyne Point Cloud
    print("CREATING DESCRIPTORS WITH VELODYNE POINTS")
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

if USE_SEM and USE_VELO and not USE_MOD and not USE_COL: # Use Semantic Velodyne Point Cloud (old way)
    print("CREATING DESCRIPTORS WITH VELODYNE-SEMANTIC POINTS, ARGMAX METHOD")
    descriptors = {}
    if resume is None:
        with open("descriptor_texts/sem_velo_descriptors_kitti_"+seq_name+".txt", 'w') as file:
            for im in trange(seq_leng):
                point_cloud = velo_proc.createCloudSem(im)
                labels = point_cloud[:,3]
                point_cloud = point_cloud[:,:3]
                descriptors[im] = createDescriptor(point_cloud)
                sem_descriptor = createSemDescriptor(point_cloud, labels, T=16, L=8)
                comp_sem_descriptor, descriptor_size = des_compress_new(sem_descriptor)
                line1 = np.array2string(descriptors[im], max_line_width=10000, separator=';')
                line2 = np.array2string(comp_sem_descriptor, max_line_width=50000, separator=';', threshold=10000)
                file.write(line1+"\n"+line2+"\n")
    else:
        with open("descriptor_texts/sem_velo_descriptors_kitti_"+seq_name+".txt", 'a') as file:
            for im in trange(resume, seq_leng):
                point_cloud = velo_proc.createCloudSem(im)
                labels = point_cloud[:,3]
                point_cloud = point_cloud[:,:3]
                descriptors[im] = createDescriptor(point_cloud)
                sem_descriptor = createSemDescriptor(point_cloud, labels)
                comp_sem_descriptor, descriptor_size = des_compress_new(sem_descriptor)
                line1 = np.array2string(descriptors[im], max_line_width=10000, separator=';')
                line2 = np.array2string(comp_sem_descriptor, max_line_width=50000, separator=';', threshold=10000)
                file.write(line1+"\n"+line2+"\n")
    print(len(descriptors))
    print(descriptor_size)

if USE_MOD and USE_VELO and not USE_SEM and not USE_COL: # Use Constrained Velodyne Point Cloud
    print("CREATING DESCRIPTORS WITH VELODYNE POINTS, CONSTRAINED VIEW")
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

if USE_COL and USE_VELO and not USE_SEM and not USE_MOD: # Color Only Descriptor
    print("CREATING DESCRIPTORS WITH VELODYNE-COLOR POINTS")
    descriptors = {}
    if resume is None:
        with open("descriptor_texts/color_descriptors_kitti_"+seq_name+".txt", 'w') as file:
            for im in trange(seq_leng):
                point_cloud = velo_proc.createCloudFull(im)
                point_cloud = np.delete(point_cloud, 3, 1)
                descriptors[im] = createColorDescriptor(point_cloud)
                line = np.array2string(descriptors[im], max_line_width=50000, separator=';')
                file.write(line+"\n")
    else:
        with open("descriptor_texts/color_descriptors_kitti_"+seq_name+".txt", 'a') as file:
            for im in trange(resume, seq_leng):
                point_cloud = velo_proc.createCloudFull(im)
                point_cloud = np.delete(point_cloud, 3, 1)
                descriptors[im] = createColorDescriptor(point_cloud)
                line = np.array2string(descriptors[im], max_line_width=50000, separator=';')
                file.write(line+"\n")
    
    print(len(descriptors))

if USE_SEM and USE_VELO and USE_MOD and not USE_COL: # Use Semantic Velodyne Point Cloud with different descriptor
    print("CREATING DESCRIPTORS WITH VELODYNE-SEMANTIC POINTS, HISTOGRAM METHOD")
    descriptors = {}
    if resume is None:
        with open("descriptor_texts/sem_histo_velo_descriptors_kitti_"+seq_name+".txt", 'w') as file:
            for im in trange(seq_leng):
                point_cloud = velo_proc.createCloudSem(im)
                labels = point_cloud[:,3]
                point_cloud = point_cloud[:,:3]
                descriptors[im] = createSemDescriptorHisto(point_cloud, labels)
                line1 = np.array2string(descriptors[im], max_line_width=50000, separator=';', threshold=10000)
                file.write(line1+"\n")
    else:
        with open("descriptor_texts/sem_histo_velo_descriptors_kitti_"+seq_name+".txt", 'a') as file:
            for im in trange(resume, seq_leng):
                point_cloud = velo_proc.createCloudSem(im)
                labels = point_cloud[:,3]
                point_cloud = point_cloud[:,:3]
                descriptors[im] = createSemDescriptorHisto(point_cloud, labels)
                line1 = np.array2string(descriptors[im], max_line_width=50000, separator=';', threshold=10000)
                file.write(line1+"\n")
    print(len(descriptors))
    #print(descriptor_size)