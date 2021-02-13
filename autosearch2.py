# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 22:03:38 2020

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

if len(sys.argv) > 1:
    scene_name = sys.argv[1]
    scene_path = os.path.join(root,scene_name)
    requested_instance = sys.argv[2]

instance_found = tm.check_present_instance(scene_path, requested_instance)
if not instance_found:
    print("Instance requested is not in the selected scene.")
    
else:
    #set up target id
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
    move_command = ''
    
    fig,ax = plt.subplots(1)
    approaches = 0
    turned = 0
    #found = False
    reach = False
    dead_end = False
    looped = False
    first_dead_end = ''
    
    
    while not reach:
        img = tm.plot_image(images_path, cur_image_name, annotations, ax)
        boxes = annotations[cur_image_name]['bounding_boxes']
        found = tm.search(id_num, boxes, img, ax)
        
        #SET MOVE COMMANDS FIRST:
        
        #if approach module is tried once or more, change direction
        if approaches >=1:
            next_image_name = tm.command('d', cur_image_name, annotations)
            approaches = 0
     
        #if found, enter approach module
        if found:    
            reach = tm.approach(images_path, cur_image_name, annotations, ax, id_num)
            approaches+=1
            
        else:   
            #non-repeated dead end
            if dead_end:
                #after turning a few times, go straight                    
                if turned >= 12:
                    next_image_name = tm.command('w', cur_image_name, annotations)
                    turned = 0
                    dead_end = False
                #if haven't turned enough, keep turning
                else:
                    next_image_name = tm.command('d', cur_image_name, annotations)
                    turned+=1
            
            #repeated dead end, so change direction
            elif looped:
                #after turning a few times, go straight
                if turned >= 12:
                    next_image_name = tm.command('w', cur_image_name, annotations)
                    turned = 0
                    looped = False
                #if haven't turned enough, keep turning
                else:
                    next_image_name = tm.command('a', cur_image_name, annotations)
                    turned+=1
            
            #normal circumstances, just go straight
            else:
                next_image_name = tm.command('w', cur_image_name, annotations)
                

        #SETTING NEXT IMAGE AFTER MOVE COMMAND IS DETERMINED:      
                
        
        if next_image_name != '':
            #if still turning away from dead end, keep turning
            if dead_end:
                next_image_name = tm.command('d', cur_image_name, annotations)
                cur_image_name = next_image_name
                turned+=1   
            elif looped:
                next_image_name = tm.command('a', cur_image_name, annotations)
                cur_image_name = next_image_name
                turned+=1
            else:   
                cur_image_name = next_image_name
                
        elif next_image_name == '':     
            #repeated dead end, change direction
            if cur_image_name == first_dead_end:
                looped = True
                next_image_name = tm.command('a', cur_image_name, annotations) 
                cur_image_name = next_image_name
                first_dead_end = ''
                turned+=1
                dead_end = False
                
            #not a repeated dead end    
            else:
                looped = False
                #if it's a first dead end of a loop, record image as first dead end
                if first_dead_end == '':
                    first_dead_end = cur_image_name
                    next_image_name = tm.command('d', cur_image_name, annotations) 
                    cur_image_name = next_image_name
                    turned+=1
                    dead_end = True
                #if it's not a first dead end of a loop, don't record    
                else:
                    next_image_name = tm.command('d', cur_image_name, annotations) 
                    cur_image_name = next_image_name
                    turned+=1
                    dead_end = True
                
                
        
    plt.waitforbuttonpress(-1)
        
    
                
        
