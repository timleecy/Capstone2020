# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:44:23 2020

@author: timot
"""
import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import misc
import scipy.io as sio
import cv2 as cv
import numpy as np
import test_modules as tm
    
root = 'C:/Capstone2020/ActiveVisionDataset/'
instance_ids = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,
                21,22,23,24,25,26,27,28]
all_scenes_list = [ 
                      'Home_001_1',
                      'Home_001_2',
                      'Home_002_1',
                      'Home_003_1',
                      'Home_003_2',
                      'Home_004_1',
                      'Home_004_2',
                      'Home_005_1',
                      'Home_005_2',
                      'Home_006_1',
                      'Home_008_1',
                      'Home_014_1',
                      'Home_014_2',
                      'Office_001_1'
]


if len(sys.argv) > 1:
    scene_name = sys.argv[1]
    scene_path = os.path.join(root,scene_name)
    requested_instance = sys.argv[2]

instance_found = tm.check_present_instance(scene_path, requested_instance)
if not instance_found:
    print("Instance requested is not in the selected scene.")
    
else:
    #VISUALIZATIONS
    name_to_id_dict = tm.get_instance_name_to_id_dict(root)
    id_num = name_to_id_dict[requested_instance[:]]

    #set up scene specfic paths
    images_path = os.path.join(scene_path,'jpg_rgb')
    annotations_path = os.path.join(scene_path,'annotations.json')

    #load data
    image_names = os.listdir(images_path)
    image_names.sort()
    ann_file = open(annotations_path)
    annotations = json.load(ann_file)

    #set up for first image
    cur_image_name = image_names[0]
    next_image_name = ''
    dead_end=''
    move_command = ''
    
    fig,ax = plt.subplots(1)
    found = False
    reach = False
    while not reach:
        
        
        #load the current image and annotations 
        rgb_image = cv.imread(os.path.join(images_path,cur_image_name))
        boxes = annotations[cur_image_name]['bounding_boxes']
        
        plt.cla()
        ax.imshow(rgb_image)
        plt.title(cur_image_name)
        ax.axis('off')
        
        for box in boxes:
            if box[4]==id_num:
                
                # Create a Rectangle patch
                rect =patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                                         linewidth=2,edgecolor='r',facecolor='none')
    
                #makes sure object is in center
                found = tm.center(rect,rgb_image)
                
                # Add the patch to the Axes
                #if found==True:
                ax.add_patch(rect)
                
                
        #draw the plot on the figure
        plt.draw()
        plt.pause(0.01)
        
        
        if found:
            next_image_name = tm.command('w', cur_image_name, annotations)
            if next_image_name == '':
                plt.waitforbuttonpress(-1)
                reach = True
        else:
            if not dead_end:
                next_image_name = tm.command('w', cur_image_name, annotations)
            else:
                next_image_name = tm.command('a', cur_image_name, annotations)
            
            
        #if the user inputted move is valid (there is an image there) 
        #then update the image to display. If the move was not valid, 
        #the current image will be displayed again
        if next_image_name != '':
            cur_image_name = next_image_name
        else:
            dead_end = cur_image_name
            #if found==True:
             #   cur_image_name = next_image_name
            #else:
            next_image_name = tm.command('d', cur_image_name, annotations)
            cur_image_name = next_image_name
            
            

            
        
            
                







