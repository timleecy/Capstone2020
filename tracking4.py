# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:35:07 2020

@author: timot
"""

import sys
import os
import numpy  as np
import json
import matplotlib.pyplot as plt
import cv2 as cv
from project_modules2 import *
import math
import random



#set path and user inputs
root = 'C:/Capstone2020/ActiveVisionDataset/'
scene_name = sys.argv[1]
scene_path = os.path.join(root,scene_name)
target_colour = sys.argv[2]
target_shape = sys.argv[3]
    
#set up scene specific paths
images_path = os.path.join(scene_path,'jpg_rgb')
annotations_path = os.path.join(scene_path,'annotations.json')
depth_path = os.path.join(scene_path, 'high_res_depth')
    
#load data
image_names = os.listdir(images_path)
image_names.sort()
ann_file = open(annotations_path)
annotations = json.load(ann_file)
depth_names = os.listdir(depth_path)
depth_names.sort()

#set up for first image
cur_image_name = image_names[0]
next_image_name = '' 
cur_depth_name = depth_names[0]

#set up tracking plot
x_co = 0
y_co = 0
dx = 0
dy = 0
theta = -30

#start main loop
fig,ax = plt.subplots()     
object_area = []
found = False
done = False
times_detected = 0
waited = False
cur_obj_area = 0
prev_obj_area = None
found_obj_coordinates = []
start = True
prev_move = 119
while times_detected<7:
    img = cv.imread(os.path.join(images_path,cur_image_name))
    
    
    if target_shape == 'circle':
        found, img = search_circle(img, target_colour)
    else:
        found, img, mask, found_object = search_polygon(img, target_shape, target_colour)
        
    '''if found:
        cur_obj_area = get_area(mask, found_object, depth_path, cur_depth_name)
        
        if math.isnan(cur_obj_area):
            cur_obj_area = prev_obj_area
        
        if prev_obj_area is not None:
            similar = compare_area(cur_obj_area, prev_obj_area)
 
            if similar:
                prev_obj_area = cur_obj_area
                times_detected += 1
                waited = False
            else:
                prev_obj_area = cur_obj_area
                times_detected = 0
                waited = False
        else:
            if waited:
                prev_obj_area = None
                times_detected = 0
                waited = False
            else:
                prev_obj_area = cur_obj_area
                waited = True
    else:
        if waited:
                prev_obj_area = None
                times_detected = 0
                waited = False
        else:
            prev_obj_area = cur_obj_area
            waited = True'''
    
    #consecutive counter      
    if found:
        times_detected += 1
        waited = False
    else: 
        if waited:
            times_detected = 0
            waited = False
        else:
            waited = True
    print(times_detected)
 
    key, x_co, y_co = display(x_co, y_co, dx, dy, found, fig, img)
    
    '''if found:
        found_obj_coordinates.append([x_co,y_co])'''
    
    if key == 113:
        cv.destroyAllWindows
        break
    else:
        if start:
            key = 100 #rotate clockwise
            next_image_name = command(key, annotations, cur_image_name)
            start = False
        else:
            if found:
                #print(found_object)
                
                '''if times_detected == 1:
                    
                    if found_object[0][0] >= 0.8*img.shape[1]/2:
                        #key = 114
                        key = 100
                        next_image_name = command(key, annotations, cur_image_name)
                        
                    elif found_object[0][0] <= 0.2*img.shape[1]/2:
                        #key = 101
                        key = 97
                        next_image_name = command(key, annotations, cur_image_name)
                        
                    else:
                        key = 119
                        next_image_name = command(key, annotations, cur_image_name)
                
                else:'''
                if found_object[0][0] >= 0.7*img.shape[1]/2:
                    key = 114
                    prev_move = key
                    #key = 100
                    next_image_name = command(key, annotations, cur_image_name)
                    
                elif found_object[0][1] <= 0.3*img.shape[1]/2:
                    key = 101
                    prev_move = key
                    #key = 97
                    next_image_name = command(key, annotations, cur_image_name)
                   
                else:
                    key = 119
                    prev_move = key
                    next_image_name = command(key, annotations, cur_image_name)
        
            else:
                    key = 119 #forward
                    next_image_name = command(key, annotations, cur_image_name)
                
        
        #if dead end, rotate clockwise
        if next_image_name == '':
            key = random.choice([97, 100]) 
            next_image_name = command(key, annotations, cur_image_name)
            #next_image_name = command(key, annotations, next_image_name)
            
        
    
    cur_image_name, cur_depth_name, dx, dy, theta = update_pos_img(cur_image_name, next_image_name, key, theta)
   
    
#print(found_obj_coordinates)    
        
cv.waitKey(-1)       


