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
turned = False
prev_key = random.choice([97, 100]) 
dead_end = False
rotation_counter = 0
steps_moved = 0
prev_found_img = ''

while times_detected<5:
    img = cv.imread(os.path.join(images_path,cur_image_name))
    
    if target_shape == 'circle':
        found, img = search_circle(img, target_colour)
    else:
        found, img, mask, found_object = search_polygon(img, target_shape, target_colour, depth_path, cur_depth_name)
     
     
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
        
    
    if start:
        key = 100 #rotate clockwise
        next_image_name = command_disp(key, annotations, cur_image_name, img)
        start = False
    else:
        if found:
            x, y, w, h = cv.boundingRect(found_object)
            cx = x + w/2
            
            if cx >= 0.75*img.shape[1]:
                key = 100
                next_image_name = command_disp(key, annotations, cur_image_name, img)

            elif cx <= 0.25*img.shape[1]:
                key = 97
                next_image_name = command_disp(key, annotations, cur_image_name, img)
                
            else:
                key = 119
                next_image_name = command_disp(key, annotations, cur_image_name, img)
                
        else:
            key = 119 #forward
            next_image_name = command_disp(key, annotations, cur_image_name, img)
            
        
    if next_image_name != '':
        cur_image_name = next_image_name
    else:
        key = random.choice([97,100]) 
        next_image_name = command_disp(key, annotations, cur_image_name, img)
        cur_image_name = next_image_name
    
        
cv.waitKey(-1)       


