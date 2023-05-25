# Class Based implementation of the Segmentor Code Written Previously

# ----- Package Imports -----
import torch
import numpy as np
import cv2 as cv
import yaml
import csv
from scipy.io import loadmat
from yaml.loader import SafeLoader
import torchvision.transforms

# Import Required MIT_Semseg Packages
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

# ----- Class Definition ----- 
class SegmentationEngine:
    """
    Initialises a SegmentationEngine Object, which is used to perform semantic segmentation on images 

    Parameters:
    ----------
        model_id: Defines which Segmentation Model to use (refer to model_info.yaml for IDs).
                    Defaults value: 6
        
        use_gpu: Defines whether tensor operations should be processed using the discrete GPU.
                    Default value: False

    Instance Variables:
    ----------
        use_gpu: Boolean
        enc: String
        dec: String

        colours: Dict
        names: Dict

        segmentation_module: SegmentationModule object
                    
    Returns:
    ----------
        Instance of SegmentationEngine object
    """
    def __init__(self, model_id=1, use_gpu=False):
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

        self.colours = loadmat(col_path)['colors']
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

        print(model_data['enc_arch'], model_data['dec_arch'])

    # ----- Method Definitions -----
    def visualise_result(self, img, pred, index=None):
        """
        Creates a visualisation for a given segmentation result, using the original image and the matrix of predictions for each pixel.
        The predictions define the colours used for visualisation, based on a set of pre-defined colours. 

        Parameters:
        ----------
            img: Matrix containing the loaded image
            
            pred: Matrix of pixel-wise class predictions for each pixel in img

            index: Specifies specific prediction class to visualise in resultant image.
                    Default value: None

        Returns:
        ----------
            output: Image matrix of same size as img
        """
        if index is not None:
            pred = pred.copy()
            pred[pred != index] = -1
            print(f'{self.names[index+1]}:')
        pred_color = colorEncode(pred, self.colours).astype(np.uint8)
        output = ((0.3 * img) + (0.7 * pred_color)).astype(np.uint8)
        return output

    def segmentImageVis(self, img):
        """
        Segments a given image, and then generates a visualisation of that image based on the predictions of the
        semantic segmentation 

        Parameters:
        ----------
            img: Matrix containing the loaded image

        Returns:
        ----------
            output: Image matrix of same size as img, with added colour overlay showing semantic classes
        """
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
        """
        Semantically segments a given image, and then returns the full class distribution tensor for all classes across all pixels in the image. 

        Parameters:
        ----------
            img: Matrix containing the loaded image

        Returns:
        ----------
            dist: Tensor of probability of each pixel being a specific class
        """
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
        """
        Semantically segments a given image, and then returns only the maximum likelihood class for each pixel in the image

        Parameters:
        ----------
            img: Matrix containing the loaded image

        Returns:
        ----------
            pred: Matrix of Maximum Likelihood classes for each pixel
        """
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
    img_col = cv.imread("Test_Image.png")
    img_col = cv.cvtColor(img_col, cv.COLOR_BGR2RGB)
    #img_col = cv.GaussianBlur(img_col, (1,1), 0)

    img_bw = cv.imread("Test_BW.png")
    img_bw = cv.cvtColor(img_bw, cv.COLOR_BGR2RGB)
    #img_bw = cv.GaussianBlur(img_bw, (1,1), 0)

    segEngine = SegmentationEngine(model_id=2, use_gpu=False)

    output_col = segEngine.segmentImageVis(img_col)
    output_bw = segEngine.segmentImageVis(img_bw)
    #dist = segEngine.segmentImageDist(img_col)
    #max = segEngine.segmentImageMax(img_col)

    output_col = cv.cvtColor(output_col, cv.COLOR_RGB2BGR)
    output_bw = cv.cvtColor(output_bw, cv.COLOR_RGB2BGR)

    cv.imshow("Segmentation Result - Colour", output_col)
    cv.imshow("Segmentation Result - B&W", output_bw)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #print(output.shape, output.dtype)
    #print(dist.shape, dist.dtype)
    #print(max.shape, max.dtype)