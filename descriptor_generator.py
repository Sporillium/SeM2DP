# Code for evaluating different loop closure methods against ground truth

# Custom File Imports:
import segmentation as seg
import stereo
import cloud
import m2dp

# Python Package imports
import numpy as np
from tqdm import trange

# Define Execution flags:
SEQ_NUM = 7

# ----- Main Execution Starts here -----
# Load processing objects:
segmentation_engine = seg.SegmentationEngine(model_id=6, use_gpu=False)
stereo_extractor = stereo.StereoExtractor(segmentation_engine, detector='SIFT', matcher='BF', camera_id=0, seq=SEQ_NUM)
cloud_engine = cloud.CloudProcessor(stereo_extractor)
signature_generator = m2dp.m2dp()

seq_leng = stereo_extractor.seq_len
seq_name = f'{SEQ_NUM:02}'

descriptors = {}
with open("descriptor_texts/basic_descriptors_kitti_"+seq_name+".txt", 'w') as file:
    for im in trange(seq_leng):
        point_cloud = cloud_engine.processFrameNoSemantics(im)
        descriptors[im] = signature_generator.extractAndProcess(point_cloud)
        line = np.array2string(descriptors[im], max_line_width=10000, separator=';')
        file.write(line+"\n")

print(len(descriptors))
