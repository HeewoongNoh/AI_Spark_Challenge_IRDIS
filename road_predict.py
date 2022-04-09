import argparse
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from PIL import Image

from networks.dinknet import LinkNet34, DinkNet34
from networks.nllinknet_location import NL3_LinkNet, NL4_LinkNet, NL34_LinkNet, Baseline
from networks.nllinknet_pairwise_func import NL_LinkNet_DotProduct, NL_LinkNet_Gaussian, NL_LinkNet_EGaussian
from networks.unet import Unet
from util.test_framework import TTAFramework

import ssl
ssl._create_default_https_context = ssl._create_unverified_context #for remote development. not needed if on main server


'''
1 generative model for roads: tif_to_png()
2 testing models for roads: test_models(), test_models_imageoutput()
file requirements: have xview2_data in the same running directory as this .py file.
xview_data should have 2 sub folders called post_damaged, pre_damaged, containing raw tif files.
'''

'''
test_models() reads all png files generated from tif_to_png() (default path is called tif_to_png/)
Predicts image with a trained NL34_LinkNet, saves (1024,1024) np.array containing class labels
for each pixel to path (default path is road_predict_array/)
'''

def test_models(model, n_class, name, source, scales=(1.0,), target=''):
    if type(scales) == tuple:
        scales = list(scales)
    print(model, name, source, scales, target)

    solver = TTAFramework(model, n_class) #Loads NL34_LinkNet from pytorch
    solver.load('weights/' + name + '.th') #Loads model weights - default nia_cls_7_500
    png_path = glob.glob(os.path.join(source, "*.png")) #Saves all png files from source(tif_to_png) to iterable list

    if target == '':
        target = 'road_predict_array/'
    else:
        target = 'road_predict_array/' + target + '/'
    os.makedirs(target, exist_ok=True) #creates directory in which the arrays are saved

    len_scales = int(len(scales))
    if len_scales > 1:
        print('multi-scaled test : ', scales)

    for i, name in tqdm(enumerate(png_path), ncols=10, desc="Testing "): #enumerates over png file names
        mask = solver.test_one_img_from_path(png_path[i], scales) #loads pytorch inference module, predicts each class, saves (1024,1024,7) array to mask
        mask = np.argmax(mask, axis=0) #argmax for each pixel, saves class index to (1024,1024) array to mask
        file_name = target+f'{os.path.basename(png_path[i][:-4])}_road_array_2dim.npy' #npy file name for mask, saves to road_predict_array/
        print(f'saving {file_name}..')
        np.save(file_name, mask)

'''
test_models_imageoutput() reads all png files generated from tif_to_png()
Predicts image with a trained NL34_LinkNet, saves 3 images (blend,mask, pred_idx)
to path (default path is road_predict_image/)

blend.png: original image+rgb prediction from 3dim array
mask.png: rgb prediction from 3dim array
pred_idx: class prediction to image from 2dim array

Colors
Mortorway: Blue
Primary: Cyan
Secondary: Green
Tertiary: Yellow
Residential: Red
Unclassified: Purple
Background: Black
'''

def test_models_imageoutput(model, n_class, name, source, scales=(1.0,), target=''):
    if type(scales) == tuple:
        scales = list(scales)
    print(model, name, source, scales, target)

    colorbook = {"Mortorway":(51, 51, 255), "Primary":(51, 255, 255), "Secondary":(51, 255, 51), "Tertiary":(255, 255, 51), "Residential":(255, 51, 51), "Unclassified":(255, 51, 255), "background":(0, 0, 0)}
    class_2_id = {"Mortorway":1, "Primary":2, "Secondary":3, "Tertiary":4, "Residential":5, "Unclassified":6, "background":0}

    solver = TTAFramework(model, n_class) #Loads NL34_LinkNet from pytorch
    solver.load('weights/' + name + '.th')
    png_path = glob.glob(os.path.join(source, "*.png")) #Saves all png files from source(tif_to_png) to iterable list

    if target == '':
        target = 'road_predict_image/'
    else:
        target = 'road_predict_image/' + target + '/'
    os.makedirs(target, exist_ok=True) #creates directory in which the images are saved

    len_scales = int(len(scales))
    if len_scales > 1:
        print('multi-scaled test : ', scales)

    for i, name in tqdm(enumerate(png_path), ncols=10, desc="Testing "): #enumerates over png file names
        mask = solver.test_one_img_from_path(png_path[i], scales) #loads pytorch inference module, predicts each class, saves (1024,1024,7) array to mask
        mask = np.argmax(mask, axis=0) #argmax for each pixel, saves class index to (1024,1024) array to mask
        img = cv2.imread(png_path[i]) #reads image for masking
        mask_im = np.zeros_like(img) #creates zerolike array for image masking
        alpha = 0.2 #color weight

        for c in class_2_id: #iterates over 7 road classes
            if c == "background": continue #skips background
            c_id = class_2_id[c] #class id
            color = colorbook[c] #saves color tuple for class id (ex. Mortorway color is (51,51,255), or blue)
            temp = np.zeros_like(img) #creates zerolike array for temporary masking
            temp[mask == c_id] = color #saves rgb values to temporary array where the color id matches the mask. 3dim array
            mask_im[mask == c_id] = color #saves rgb values to temporary array where the color id matches the mask. 3dim array
            temp = cv2.bitwise_or(temp, img) #bitwise OR operation for temp, img array
            img = cv2.addWeighted(img, alpha, temp, 1-alpha, 0) #weight operation for img, temp array

        print(f'writing number {i} image..')
        cv2.imwrite(target + os.path.basename(png_path[i][:-4]) + '_blend.png', img)
        cv2.imwrite(target + os.path.basename(png_path[i][:-4]) + '_mask.png', mask_im)
        cv2.imwrite(target + os.path.basename(png_path[i][:-4]) + '_pred_idx.png', mask)
        file_name = target+f'{os.path.basename(png_path[i][:-4])}_road_array_3dim.npy' #npy file name for mask, saves to road_predict_array/
        print(f'saving {file_name}..')
        np.save(file_name, mask_im)
        
        print(f'writing number {i} image done!')

'''
tif_to_png() reads all tif files from tifsource (set default path to datasource)
transforms raster metadata into RGB channels and saves image to path (default path is tif_to_png/)
'''

def tif_to_png(tifsource,target):
    post_damaged_road = glob.glob(os.path.join(tifsource, "post_damaged", "*.tif"))
    pre_damaged_road = glob.glob(os.path.join(tifsource, "pre_damaged", "*.tif"))
    if target == '':
        target = 'tif_to_png/'
    else:
        target = 'tif_to_png/' + target + '/'
    os.makedirs(target, exist_ok=True)

    for i, name in enumerate(pre_damaged_road): #change to post_damaged_road for post
        road_tif = gdal.Open(name) #opens tif file
        red = road_tif.GetRasterBand(1).ReadAsArray()
        green = road_tif.GetRasterBand(2).ReadAsArray()
        blue = road_tif.GetRasterBand(3).ReadAsArray()
        rgbOutput = np.zeros((1024,1024,3))
        rgbOutput[...,2] = red #saves each 2dim array to 3dim array
        rgbOutput[...,0] = green
        rgbOutput[...,1] = blue
        rgbOutput = (rgbOutput * 255).astype(np.uint8)
    
        img_rgb = Image.fromarray(rgbOutput, 'RGB')
        img_rgb.save(target+os.path.basename(pre_damaged_road[i][:-4])+'_fromtif.png') #change to post_damaged_road for post



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="set model name", default="NL34_LinkNet")
    parser.add_argument("--n_class", help="Number of class output", type=int, default=7)
    parser.add_argument("--name", help="set path of weights", default="nia_cls_7_500")
    #parser.add_argument("--source", help="path of test datasets", default='Module 2/datasets/valid/valid_mini')
    parser.add_argument("--tifsource", help="path of tif datasets", default='../xview2_sample')
    parser.add_argument("--source", help = "path of png datasets", default='tif_to_png')
    parser.add_argument("--scales", help="set scales for MST", default=[1.0], type=float, nargs='*')
    parser.add_argument("--target", help="new folder name for road array files", default='')

    args = parser.parse_args()

    models = {'NL3_LinkNet': NL3_LinkNet, 'NL4_LinkNet': NL4_LinkNet, 'NL34_LinkNet': NL34_LinkNet,
              'Baseline': Baseline,
              'NL_LinkNet_DotProduct': NL_LinkNet_DotProduct, 'NL_LinkNet_Gaussian': NL_LinkNet_Gaussian,
              'NL_LinkNet_EGaussian': NL_LinkNet_EGaussian,
              'UNet': Unet, 'LinkNet': LinkNet34, 'DLinkNet': DinkNet34}

    model = models[args.model]
    name = args.name
    n_class = args.n_class
    scales = args.scales
    target = args.target
    tifsource = args.tifsource
    source = args.source

    '''
    uncomment functions for use
    '''

    tif_to_png(tifsource=tifsource, target=target)
    test_models(model=model, n_class=n_class, name=name, source=source, scales=scales, target=target)
    test_models_imageoutput(model=model, n_class=n_class, name=name, source=source, scales=scales, target=target)


if __name__ == "__main__":
    main()