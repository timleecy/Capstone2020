# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 21:44:00 2020

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

min_distance = 500
obstacle_side = []
times_detected = 0
waited = False
start = True
while times_detected < 6:
    img = cv.imread(os.path.join(images_path,cur_image_name))
    
    next_image_name = command_disp(100, annotations, cur_image_name, img) 
    cv.waitKey(5000)
    next_image_name = command_disp(100, annotations, next_image_name, img) 
                
    
    #If there is an image available, continue navigating forward
    if next_image_name != '':
        cur_image_name = next_image_name
        
    
    
            
            
cv.waitKey(-1)           
        
        
    
    