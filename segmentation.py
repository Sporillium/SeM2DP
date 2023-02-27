# Class Based implementation of the Segmentor Code Written Previously

# ----- Package Imports -----
import torch
import numpy as np
import cv2 as cv
import yaml
import csv
import scipy.io
from yaml.loader import SafeLoader
import torchvision.transforms

# Import Required MIT_Semseg Packages
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

class SegmentationEngine:
    def __init__(self, model_id=6, use_gpu=False):
        self.use_gpu = use_gpu
        # Read YAML file with required config options in it:
        with open('./model_info/model_info.yaml') as f:
            config = yaml.load(f, Loader=SafeLoader)
            col_path = config['colors']
            names_path = config['names']
            all_data = config['ID']
            model_data = all_data[model_id]
            #print(model_data['enc_arch'], model_data['dec_arch'])
            self.enc = model_data['enc_arch']
            self.dec = model_data['dec_arch']
            f.close()

        self.colours = scipy.io.loadmat(col_path)['colors']
        self.names = {}
        with open(names_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.names[int(row[0])] = row[5].split(";")[0]

        net_encoder = ModelBuilder.build_encoder(
            arch=model_data['enc_arch'],
            fc_dim=model_data['fc_dim'],
            weights=model_data['enc_weights'])

        net_decoder = ModelBuilder.build_decoder(
            arch=model_data['dec_arch'],
            fc_dim=model_data['fc_dim'],
            num_class=model_data['num_classes'],
            weights=model_data['dec_weights'],
            use_softmax=model_data['use_softmax'])

        crit = torch.nn.NLLLoss(ignore_index=-1)
        self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        self.segmentation_module.eval()
        if self.use_gpu:
            self.segmentation_module.cuda()


    # ----- Method Definitions -----
    def visualise_result(self, img, pred, index=None):
        if index is not None:
            pred = pred.copy()
            pred[pred != index] = -1
            print(f'{self.names[index+1]}:')
        pred_color = colorEncode(pred, self.colours).astype(np.uint8)
        output = ((0.3 * img) + (0.7 * pred_color)).astype(np.uint8)
        return output

    def segmentImageVis(self, img):
        to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

        img_data = to_tensor(img)
        if self.use_gpu:
            singleton_batch = {'img_data':img_data[None].cuda()}
        else:
            singleton_batch = {'img_data':img_data[None]}
        output_size = img_data.shape[1:]

        with torch.no_grad():
            scores = self.segmentation_module(singleton_batch, segSize=output_size)

        _, pred = torch.max(scores, dim=1)
        if self.use_gpu:
            pred = pred.cpu()[0].numpy()
        else:
            pred = pred[0].numpy()

        output = self.visualise_result(img, pred)
        return output
        
    def segmentImageDist(self, img):
        to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

        img_data = to_tensor(img)
        if self.use_gpu:
            singleton_batch = {'img_data':img_data[None].cuda()}
        else:
            singleton_batch = {'img_data':img_data[None]}
        output_size = img_data.shape[1:]

        with torch.no_grad():
            scores = self.segmentation_module(singleton_batch, segSize=output_size)
        if self.use_gpu:
            dist = scores.cpu()[0].numpy()
        else:
            dist = scores[0].numpy()

        return dist

    def segmentImageMax(self, img):
        to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

        img_data = to_tensor(img)
        if self.use_gpu:
            singleton_batch = {'img_data':img_data[None].cuda()}
        else:
            singleton_batch = {'img_data':img_data[None]}
        output_size = img_data.shape[1:]

        with torch.no_grad():
            scores = self.segmentation_module(singleton_batch, segSize=output_size)

        _, dist = torch.max(scores, dim=1)
        if self.use_gpu:
            pred = dist.cpu()[0].numpy()
        else:
            pred = dist[0].numpy()
        
        return pred.astype(np.int32)


# ----- Execution if Main -----
if __name__ == '__main__':
    img = cv.imread("Test_Image.png")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    segEngine = SegmentationEngine(model_id=1, use_gpu=False)

    output = segEngine.segmentImageVis(img)
    dist = segEngine.segmentImageDist(img)
    max = segEngine.segmentImageMax(img)

    output = cv.cvtColor(output, cv.COLOR_RGB2BGR)

    cv.imshow("Segmentation Result", output)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print(output.shape, output.dtype)
    print(dist.shape, dist.dtype)
    print(max.shape, max.dtype)