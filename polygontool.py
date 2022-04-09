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

'''
creates polygon pngs from only json files.
'''


def polygon_road():
    DIR_geojsons = 'Module 2/datasets/valid/valid_mini/label/'
    DIR_save = 'polygon_road/'
    image_size = 1024
    GTformat = False #True: saves array as 3 dimensional array with RGB values. False: saves array as 2 dimensional array with class id values.

    os.makedirs(DIR_save, exist_ok=True)

    list_geojson = os.listdir(DIR_geojsons)

    data = {'objects':[], 'typeid':[]}
    for i in range(list_geojson.__len__()):
        temp = pd.read_json(DIR_geojsons + list_geojson[i], orient='recods')
        buildins_per_img = []
        typeid_of_building = []
        for j in range(temp.shape[0]):
            buildins_per_img.append(temp.values[j][0]['properties']['road_imcoords'])
            typeid_of_building.append(temp.values[j][0]['properties']['type_id'])

        data['objects'].append(buildins_per_img)
        data['typeid'].append(typeid_of_building)

    starting_index = 0

    if GTformat == True:
        num_channel = 1  
    else:
        num_channel = 3

    previous_name = list_geojson[starting_index]
    GT_rgb = np.zeros((image_size, image_size, num_channel))

    cnt_empty = 0
    sum_cnt_m = np.zeros((15,1))

    for i in range(starting_index, len(list_geojson)):
        print(f'Number {i} object enumerating..')
        cnt_m = np.zeros((15,1))

        if previous_name != list_geojson[i]: 
            previous_name = list_geojson[i]
            GT_rgb = np.zeros((image_size, image_size, num_channel))

        temp = data['objects'][i]

        if temp =='EMPTY':
            cnt_empty = cnt_empty + 1
            raise Exception("This file is not builing GT file.")
        elif (temp == []):
            cnt_empty = cnt_empty + 1
            print('temp is empty')
            cv2.imwrite(DIR_save + previous_name.split('.')[0] + '.png', GT_rgb)
        else:
            for j in range(temp.__len__()): 
                temp_onepoly = temp[j].split(',')
                polygons = np.zeros((int(temp_onepoly.__len__()/2), 2), np.int32)
                for q in range(int(temp_onepoly.__len__()/2)):
                    polygons[q, 0] = float(temp_onepoly[q*2]) 
                    polygons[q, 1] = float(temp_onepoly[q*2+1])
                
                polygons = np.array(polygons)
                temp_id = int(data['typeid'][i][j])

                ''' count '''     
                for q in range(15):
                    if temp_id == q+1:
                        cnt_m[q] = cnt_m[q] + 1
                        sum_cnt_m[q] = sum_cnt_m[q] + 1

                ''' Value '''

                if polygons.__len__() > 0:
                    if GTformat ==  True:
                        if temp_id == 1: # Mortorway 
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (1))
                        elif temp_id == 2: # Primary
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (2))
                        elif temp_id == 3: # Secondary
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (3))
                        elif temp_id == 4: # Tertiary
                            GT_rgb = cv2.fillConvexPoly(GT_rgb, [polygons], (4))
                        elif temp_id == 5: # Residential
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (5))
                        elif temp_id == 6: # Unclassified
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (6))
                        else:
                            _=i

                    else:
                        if temp_id == 1: # Mortorway 
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (51, 51, 255))
                        elif temp_id == 2: # Primary
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (51, 255, 0))
                        elif temp_id == 3: # Secondary
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (51, 255, 51))
                        elif temp_id == 4: # Tertiary
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (255, 255, 51))
                        elif temp_id == 5: # Residential
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (255, 51, 51))
                        elif temp_id == 6: # Unclassified
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (255, 51, 255))
                        else:
                            _=i

            cv2.imwrite(DIR_save + previous_name.split('.')[0] + '_polygon_road' + '.png', GT_rgb) # save image


def polygon_building():
    DIR_geojsons = 'jsonpolygontest/'
    DIR_save = 'polygon_building/'
    image_size = 1024
    GTformat = False #True: saves array as 3 dimensional array with RGB values. False: saves array as 2 dimensional array with class id values.

    os.makedirs(DIR_save, exist_ok=True)

    list_geojson = os.listdir(DIR_geojsons)

    data = {'objects':[], 'typeid':[]}
    for i in range(list_geojson.__len__()):
        temp = pd.read_json(DIR_geojsons + list_geojson[i], orient='recods')
        buildins_per_img = []
        typeid_of_building = []
        for j in range(temp.shape[0]):
            buildins_per_img.append(temp.values[j][0]['properties']['building_imcoords'])
            typeid_of_building.append(temp.values[j][0]['properties']['type_id'])

        data['objects'].append(buildins_per_img)
        data['typeid'].append(typeid_of_building)

    starting_index = 0

    if GTformat == True:
        num_channel = 1  
    else:
        num_channel = 3

    previous_name = list_geojson[starting_index]
    GT_rgb = np.zeros((image_size, image_size, num_channel))

    cnt_empty = 0
    sum_cnt_m = np.zeros((15,1))

    for i in range(starting_index, len(list_geojson)):
        print(f'Number {i} object enumerating..')
        cnt_m = np.zeros((15,1))

        if previous_name != list_geojson[i]: 
            previous_name = list_geojson[i]
            GT_rgb = np.zeros((image_size, image_size, num_channel))

        temp = data['objects'][i]

        if temp =='EMPTY':
            cnt_empty = cnt_empty + 1
            raise Exception("This file is not builing GT file.")
        elif (temp == []):
            cnt_empty = cnt_empty + 1
            print('temp is empty')
            cv2.imwrite(DIR_save + previous_name.split('.')[0] + '.png', GT_rgb)
        else:
            for j in range(temp.__len__()): 
                temp_onepoly = temp[j].split(',')
                polygons = np.zeros((int(temp_onepoly.__len__()/2), 2), np.int32)
                for q in range(int(temp_onepoly.__len__()/2)):
                    polygons[q, 0] = float(temp_onepoly[q*2])
                    polygons[q, 1] = float(temp_onepoly[q*2+1])
                
                polygons = np.array(polygons)
                temp_id = int(data['typeid'][i][j])

                ''' count '''     
                for q in range(15):
                    if temp_id == q+1:
                        cnt_m[q] = cnt_m[q] + 1
                        sum_cnt_m[q] = sum_cnt_m[q] + 1

                ''' Value '''

                if polygons.__len__() > 0:
                    if GTformat ==  True:
                        if temp_id == 1: # 소형시설 
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (1))
                        elif temp_id == 2: # 아파트
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (2))
                        elif temp_id == 3: # 공장
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (3))
                        elif temp_id == 4: # 컨테이너
                            GT_rgb = cv2.fillConvexPoly(GT_rgb, [polygons], (4))
                        elif temp_id == 5: # 중형단독시설
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (5))
                        elif temp_id == 6: # 대형시설
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (6))
                        else:
                            _=i

                    else:
                        if temp_id == 1: # 소형시설 
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (51, 51, 255))
                        elif temp_id == 2: # 아파트
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (51, 255, 0))
                        elif temp_id == 3: # 공장
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (51, 255, 51))
                        elif temp_id == 4: # 컨테이너
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (255, 255, 51))
                        elif temp_id == 5: # 중형단독시설
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (255, 51, 51))
                        elif temp_id == 6: # 대형시설
                            GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (255, 51, 255))
                        else:
                            _=i
            
            print(GT_rgb.shape)
            cv2.imwrite(DIR_save + previous_name.split('.')[0] + '_polygon_building' + '.png', GT_rgb)


def polygon_fromarray(): #works just like cv2.imread method, but with customizable RGB
    DIR_array = 'road_array/'
    DIR_save = 'road_array/'
    list_array = os.listdir(DIR_array)
    print(list_array)
    for i in range(len(list_array)):
        array_name = list_array[i][:-4]
        
        print(array_name)
        array_temp = np.load(DIR_array + array_name + '.npy')
        array_1 = array_temp/(array_temp.max()/255.0)
        array_2 = array_temp/(array_temp.max()/355.0)
        array_3 = array_temp/(array_temp.max()/455.0)
        array = np.stack((array_1,array_2,array_3), axis=-1)
        
        cv2.imwrite(DIR_save+array_name+'_fromarray.png', array)
        

        



if __name__ == '__main__':
    polygon_road()
    polygon_building()
    polygon_fromarray()

