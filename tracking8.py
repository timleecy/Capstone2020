# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:02:22 2020

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
root = 'C:\Capstone2020\ActiveVisionDataset'
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
avoided = 0
cur_obj_area = 0
prev_obj_area = None
left_rotation = 0
right_rotation = 0
stuck = 8

key = 0
while True:
    img = cv.imread(os.path.join(images_path,cur_image_name))
    cur_depth_name = cur_image_name[0:13] + '03.png'
    depth_image =  cv.imread(os.path.join(depth_path, cur_depth_name), cv.IMREAD_ANYDEPTH)
    #key = cv.waitKey(300)
    
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
    
    
    obstacle_side = obs_detect(depth_image, min_distance)
    #print(left_rotation)
    #print(right_rotation)
    
    #restore distance threshold to original if altered
    if min_distance < 500:
        min_distance = 500
    
    if start:
        key = 100
        #next_image_name, next_img = command_disp(key, annotations, images_path, cur_image_name, img)
        start = False
    else:
        
        if found:
            x, y, w, h = cv.boundingRect(found_object)
            cx = x + w/2
            
            if cx >= 0.7*img.shape[1]:
                key = 100
                next_image_name, next_img = command_disp(key, annotations, images_path, cur_image_name, img)
                
            elif cx <= 0.3*img.shape[1]:
                key = 97
                next_image_name, next_img = command_disp(key, annotations, images_path, cur_image_name, img)
                
            else:
                key = 119
                next_image_name, next_img = command_disp(key, annotations, images_path, cur_image_name, img)
                
            if times_detected == 4:
                break
        
                
        else:
            if obstacle_side[0]:
                #shifts towards the right side
                next_image_name, next_img = command_disp(115, annotations, images_path, cur_image_name, img) #backwards
                if next_image_name == '':
                    next_image_name = cur_image_name
                    next_image_name, next_img = command_disp(100, annotations, images_path, cur_image_name, img) #rotate cw
                    right_rotation += 1
                    
                elif next_image_name != '':
                    next_image_name, next_img = command_disp(100, annotations, images_path, cur_image_name, img) #rotate cw
                    right_rotation += 1
                
                #reduce distance threshold
                if left_rotation+right_rotation >= stuck:
                    print('stuck')
                    min_distance = 200
     
                    
            #else if right partition has lower threshold
            elif obstacle_side[2]:
                #shifts towards the left side
                next_image_name, next_img = command_disp(115, annotations, images_path, cur_image_name, img) #backwards
                if next_image_name == '':
                    next_image_name = cur_image_name
                    next_image_name, next_img = command_disp(97, annotations, images_path, cur_image_name, img) #rotate ccw
                    left_rotation += 1
                    
                elif next_image_name != '':
                    next_image_name, next_img = command_disp(97, annotations, images_path, cur_image_name, img) #rotate ccw
                    left_rotation += 1
                   
                
            #else if middle partition has lower threshold
            elif obstacle_side[1]:
                #shift towards the left or the right side by random
                choose_dir = random.randint(0,1)
                if choose_dir == 0:
                    next_image_name, next_img = command_disp(115, annotations, images_path, cur_image_name, img) #backwards
                    if next_image_name == '':
                        next_image_name = cur_image_name
                        next_image_name, next_img = command_disp(97, annotations, images_path, cur_image_name, img) #rotate ccw
                        next_image_name, next_img = command_disp(97, annotations, images_path, next_image_name, next_img) #rotate ccw
                        #next_image_name, next_img = command_disp(97, annotations, images_path, next_image_name, next_img) #rotate ccw
                    else:
                        next_image_name, next_img = command_disp(97, annotations, images_path, cur_image_name, img) #rotate ccw
                        next_image_name, next_img = command_disp(97, annotations, images_path, next_image_name, next_img) #rotate ccw
                        next_image_name, next_img = command_disp(97, annotations, images_path, next_image_name, next_img) #rotate ccw
                        
                elif choose_dir == 1:
                    next_image_name, next_img = command_disp(115, annotations, images_path, cur_image_name, img) #backwards
                    if next_image_name == '':
                        next_image_name = cur_image_name
                        next_image_name, next_img = command_disp(100, annotations, images_path, cur_image_name, img) #rotate ccw
                        next_image_name, next_img = command_disp(100, annotations, images_path, next_image_name, next_img) #rotate ccw
                        #next_image_name, next_img = command_disp(100, annotations, images_path, next_image_name, next_img) #rotate ccw
                    else:
                        next_image_name, next_img = command_disp(97, annotations, images_path, cur_image_name, img) #rotate ccw
                        next_image_name, next_img = command_disp(97, annotations, images_path, next_image_name, next_img) #rotate ccw
                        next_image_name, next_img = command_disp(97, annotations, images_path, next_image_name, next_img) #rotate ccw
            
                
            else:
                next_image_name, next_img = command_disp(119, annotations, images_path, cur_image_name, img) #forwards
                
                #reset rotation counter
                if left_rotation !=0 or right_rotation != 0:
                    left_rotation = 0
                    right_rotation = 0
    
               
            
            
                    
                  
        
        #If there is an image available, continue navigating forward
        if next_image_name != '':
            cur_image_name = next_image_name
            
        elif next_image_name == '':
            #shift towards the left or the right side by random
            choose_dir = random.randint(0,1)
            if choose_dir == 0:
                cur_image_name, img = command_disp(97, annotations, images_path, cur_image_name, img) #rotate ccw
                cur_image_name, img = command_disp(97, annotations, images_path, cur_image_name, img) #rotate ccw
                cur_image_name, img = command_disp(97, annotations, images_path, cur_image_name, img) #rotate ccw
                cur_image_name, img = command_disp(97, annotations, images_path, cur_image_name, img) #rotate ccw
            elif choose_dir == 1:
                cur_image_name, img = command_disp(100, annotations, images_path, cur_image_name, img) #rotate cw
                cur_image_name, img = command_disp(100, annotations, images_path, cur_image_name, img) #rotate cw
                cur_image_name, img = command_disp(100, annotations, images_path, cur_image_name, img) #rotate cw
                cur_image_name, img = command_disp(100, annotations, images_path, cur_image_name, img) #rotate cw
                        
        
        if key == 113:
             break
             cv2.destroyAllWindows()


cv.waitKey(-1)
           
        
        
    
    