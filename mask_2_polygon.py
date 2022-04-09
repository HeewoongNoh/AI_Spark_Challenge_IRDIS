import os
from re import L
from subprocess import list2cmdline
import sys

import numpy as np
import pandas as pd
# import geopandas as gpd
from skimage.draw import polygon as skpoly
# import torch
import cv2
import json
import matplotlib.pyplot as plt
import csv
from PIL import Image
import glob
from shapely import wkt
from pathlib import Path
from shapely.geometry import shape, Point, Polygon, LineString
from skimage import measure

def mask_2_polygon(array_list, min_poly_area=20*20):

    for i, file_name in enumerate(array_list):
        print(f'array number {i}')
        array = np.load(array_list[i])

        for j in range(7):
            print(f'array number {i}, class {j}')
            return_list = []
            arr = np.where(array == j, 255, 0) #3, 255, 0
            for found_obj in measure.find_contours(arr, 1.0):
                try:
                    poly = Polygon(found_obj).simplify(1)
                    if poly.is_valid and poly.area > min_poly_area:
                        return_list.append(poly)
                except ValueError:
                    pass
            print(return_list)
    
def main():
    array_path = 'Module 2/road_predict_array'
    array_list = glob.glob(os.path.join(array_path, '*.npy'))
    mask_2_polygon(array_list)

if __name__ == '__main__':
    main()