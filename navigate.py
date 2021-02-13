# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 22:06:23 2020

@author: timot
"""


import cv2 as cv
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


root = 'C:/Capstone2020/ActiveVisionDataset/'
scene_name = sys.argv[1]
scene_path = os.path.join(root,scene_name)
    
#set up scene specfic paths
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
next_depth_name = ''
move_command = ''

#set up tracking plot
x_axis=0
y_axis=0
dx=0
dy=0
theta=-30

fig,ax = plt.subplots()     
objects = np.array([])
found = False
step=0
while True:

    img = cv.imread(os.path.join(images_path,cur_image_name))
    
    plt.xlim(-50,50)
    plt.ylim(-50,50) 
    plt.arrow(x_axis, y_axis, dx, dy)  
    x_axis += dx
    y_axis += dy
    found = False
    #plt.show(block = False)
    #plt.pause(0.001)
    fig.canvas.draw()
    graph = np.array(fig.canvas.renderer.buffer_rgba())
    graph = cv.cvtColor(graph, cv.COLOR_RGBA2RGB)
    graph = cv.resize(graph, (graph.shape[1],img.shape[0]))
    
    combinedimg = np.hstack((img, graph))
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    window_height = int(combinedimg.shape[0]/1.5)
    window_width = int(combinedimg.shape[1]/1.5)
    cv.resizeWindow('image', (window_width, window_height))
    cv.imshow('image', combinedimg)
    key = cv.waitKey(1)
    
    if step == 5:
        key = cv.waitKey(-1) #walk forward 5 steps
    else:
        key = 119 #after 5 steps, quit program
    
    step+=1
    
    if key==119:
        next_image_name = annotations[cur_image_name]['forward']
    elif key==97:
        next_image_name = annotations[cur_image_name]['rotate_ccw']
    elif key==115:
        next_image_name = annotations[cur_image_name]['backward']
    elif key==100:
        next_image_name = annotations[cur_image_name]['rotate_cw']
    elif key==101:
        next_image_name = annotations[cur_image_name]['left']
    elif key==114:
        next_image_name = annotations[cur_image_name]['right']
    elif key==113:
        cv.destroyAllWindows
        break

 
    if next_image_name != '':
        cur_image_name = next_image_name
        cur_depth_name = next_image_name[0:13] + '03.png'
        
        distance = 1
        
        if key==119: #forward
            dx = distance*np.cos((90-theta)*np.pi/180)
            dy = distance*np.sin((90-theta)*np.pi/180)
            
        elif key==97: #rotate ccw
            theta -= 30
            dx = 0
            dy = 0
            
        elif key==115: #backward
            dx = distance*np.cos((90+theta)*np.pi/180)
            dy = distance*(-np.sin((90+theta)*np.pi/180))
            
        elif key==100: #rotate cw
            theta += 30
            dx = 0
            dy = 0
            
        elif key==101: #left
            dx = distance*np.cos((180-theta)*np.pi/180)
            dy = distance*np.sin((180-theta)*np.pi/180)
            
        elif key==114: #right
            dx = distance*(-np.cos((180+theta)*np.pi/180))
            dy = distance*np.sin((180+theta)*np.pi/180)
        
    else:
        dx = 0
        dy = 0


    
   
    
  
        
  

