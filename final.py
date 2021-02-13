'''
VISION-BASED NAVIGATION AND OBJECT DETECTION AT LOW FRAME RATES
MAIN PROGRAM
Prepared by:
Timothy Lee (793942) leet4@student.unimelb.edu.au
Chuah Yee Hean (794709) chuahy1@student.unimelb.edu.au
Foong Shen Hui (816355) foongs@student.unimelb.edu.au

Date: 30/10/2020
'''

import sys
import os
import json
import cv2 as cv
from final_modules import *
import random
import time
import matplotlib.pyplot as plt

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

#initialise variables
min_distance = 500
obstacle_side = []
times_detected = 0
waited = False
left_rotation = 0
right_rotation = 0
stuck = 8

#set up tracking plot
x_co=0
y_co=0
dx=0
dy=0
theta=-30
fig,ax = plt.subplots()     

start_time = time.perf_counter()
while True:
    img = cv.imread(os.path.join(images_path,cur_image_name))
    cur_depth_name = cur_image_name[0:13] + '03.png'
    depth_image =  cv.imread(os.path.join(depth_path, cur_depth_name), cv.IMREAD_ANYDEPTH)
    
    if target_shape == 'circle':
        found, img, found_object = search_circle(img, target_colour)
    else:
        found, img, found_object = search_polygon(img, target_shape, target_colour)
    
        
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
    
    
    obstacle_side = obs_detect(depth_image, min_distance)
    #restore distance threshold to original if altered
    if min_distance < 500:
        min_distance = 500
    
    
    #move to direction of object if object is detected
    if found:
        if target_shape == 'circle':
            cx = found_object[0]
        else:
            x, y, w, h = cv.boundingRect(found_object)
            cx = x + w/2
        
        if cx >= 0.7*img.shape[1]:
            next_image_name, next_img, x_co, y_co, theta, fig = command_disp(100, annotations, 
                   images_path, cur_image_name, img, x_co, y_co, theta, fig, found) #rotate cw
            
        elif cx <= 0.3*img.shape[1]:
            next_image_name, next_img, x_co, y_co, theta, fig = command_disp(97, annotations, 
                   images_path, cur_image_name, img, x_co, y_co, theta, fig, found) #rotate ccw
            
        else:
            next_image_name, next_img, x_co, y_co, theta, fig = command_disp(119, annotations, 
                   images_path, cur_image_name, img, x_co, y_co, theta, fig, found) #forwards
            
        if times_detected == 4:
            break
    
            
    else:
        #obstacle on the left
        if obstacle_side[0]:
            #shifts towards the right side
            next_image_name, next_img, x_co, y_co, theta, fig = command_disp(115, annotations, 
                   images_path, cur_image_name, img, x_co, y_co, theta, fig, found) #backwards
            
            if next_image_name == '':
                next_image_name = cur_image_name
                next_image_name, next_img, x_co, y_co, theta, fig = command_disp(100, annotations, 
                    images_path, cur_image_name, img, x_co, y_co, theta, fig, found) #rotate cw
                right_rotation += 1
                
            elif next_image_name != '':
                next_image_name, next_img, x_co, y_co, theta, fig = command_disp(100, annotations, 
                       images_path, cur_image_name, img, x_co, y_co, theta, fig, found) #rotate cw
                right_rotation += 1
            
            #reduce distance threshold
            if left_rotation+right_rotation >= stuck:
                print('stuck')
                min_distance = 200
 
                
        #obstacle on the right
        elif obstacle_side[2]:
            #shifts towards the left side
            next_image_name, next_img, x_co, y_co, theta, fig = command_disp(115, annotations, 
                   images_path, cur_image_name, img, x_co, y_co, theta, fig, found) #backwards
            
            if next_image_name == '':
                next_image_name = cur_image_name
                next_image_name, next_img, x_co, y_co, theta, fig = command_disp(97, annotations, 
                       images_path, cur_image_name, img, x_co, y_co, theta, fig, found) #rotate ccw
                left_rotation += 1
                
            elif next_image_name != '':
                next_image_name, next_img, x_co, y_co, theta, fig = command_disp(97, annotations, 
                       images_path, cur_image_name, img, x_co, y_co, theta, fig, found) #rotate ccw
                left_rotation += 1
                
            #reduce distance threshold
            if left_rotation+right_rotation >= stuck:
                print('stuck')
                min_distance = 200
               
            
        #obstacle in the middle
        elif obstacle_side[1]:
            #shift towards the left or the right side by random  
            direction = random.choice([97,100])
            next_image_name, next_img, x_co, y_co, theta, fig = command_disp(115, annotations, 
                   images_path, cur_image_name, img, x_co, y_co, theta, fig, found) #backwards
            
            if next_image_name == '':
                next_image_name = cur_image_name
                next_image_name, next_img, x_co, y_co, theta, fig = command_disp(direction, annotations, 
                                        images_path, cur_image_name, img, x_co, y_co, theta, fig, found) 
                next_image_name, next_img, x_co, y_co, theta, fig = command_disp(direction, annotations, 
                                  images_path, next_image_name, next_img, x_co, y_co, theta, fig, found) 
                
            else:
                next_image_name, next_img, x_co, y_co, theta, fig = command_disp(direction, annotations, 
                                        images_path, cur_image_name, img, x_co, y_co, theta, fig, found) 
                next_image_name, next_img, x_co, y_co, theta, fig = command_disp(direction, annotations, 
                                  images_path, next_image_name, next_img, x_co, y_co, theta, fig, found) 
        
            
        else:
            next_image_name, next_img, x_co, y_co, theta, fig = command_disp(119, annotations, 
                   images_path, cur_image_name, img, x_co, y_co, theta, fig, found) #forwards
            
            #reset rotation counter
            if left_rotation != 0 or right_rotation != 0:
                left_rotation = 0
                right_rotation = 0
                

    #If there is an image available, continue navigating forward
    if next_image_name != '':
        cur_image_name = next_image_name
        
    elif next_image_name == '':
        #shift towards the left or the right side by random
        direction = random.choice([97,100])
        for i in range(4):
            cur_image_name, img, x_co, y_co, theta, fig = command_disp(direction, annotations, 
                              images_path, cur_image_name, img, x_co, y_co, theta, fig, found) 
        

stop_time = time.perf_counter()
print('Time elapsed: ' + str(stop_time - start_time) + ' seconds')
cv.waitKey(-1)
           
        
        
    
    