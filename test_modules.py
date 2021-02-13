# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 21:56:46 2020

@author: timot
"""

import os
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



def check_present_instance(scene_path, requested_instance):
    present_instance_path = os.path.join(scene_path,'present_instance_names.txt')
    present_instances = open(present_instance_path,'r')
    present_instances = present_instances.readlines()
    
    present_instances_list = []
    for line in present_instances:
        present_instances_list.append(line.strip())
        
    present=False
    for instance in present_instances_list:
        if instance == requested_instance:
            present=True
            break
    return present


def get_instance_name_to_id_dict(root):
    """
    Returns a dict with instance names as keys and is as values
    """
    name_to_id_dict = {} 
    for line in open(os.path.join(root,'instance_id_map.txt'),'r'):
        name, id_num = str.split(line)
        name_to_id_dict[name] = int(id_num)

    return name_to_id_dict


def center(rect, img):
    x = rect.xy[0]
    y = rect.xy[1]
    row = len(img)
    col = len(img[0])
    #if 432<y<648 and 768<x<1152:
    if 0.2*row<y<0.8*row and 0.2*col<x<0.8*col:
        return True
    else:
        return False
    
    
def area_large_enough(rect):
    area = rect.get_width() * rect.get_height()
    if area>=300*100:
        return True
    else:
        return False
    
    
def command(move_command, cur_image_name, annotations):
    #get the next image name to display based on the 
    #user input, and the annotation.
   
    if move_command == 'w':
        next_image_name = annotations[cur_image_name]['forward']
    elif move_command == 'a':
        next_image_name = annotations[cur_image_name]['rotate_ccw']
    elif move_command == 's':
        next_image_name = annotations[cur_image_name]['backward']
    elif move_command == 'd':
        next_image_name = annotations[cur_image_name]['rotate_cw']
    elif move_command == 'e':
        next_image_name = annotations[cur_image_name]['left']
    elif move_command == 'r':
        next_image_name = annotations[cur_image_name]['right']
    elif move_command == 'h':
        next_image_name = cur_image_name
        print("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
              "Enter a character to move around the scene:",
              "'w' - forward", 
              "'a' - rotate counter clockwise", 
              "'s' - backward", 
              "'d' - rotate clockwise", 
              "'e' - left", 
              "'r' - right", 
              "'q' - quit", 
              "'h' - print this help menu"))
        
    return next_image_name


def plot_image(images_path, cur_image_name, annotations, ax):
    #load the current image and annotations 
    rgb_image = cv.imread(os.path.join(images_path,cur_image_name))
    plt.cla()
    ax.imshow(rgb_image)
    plt.title(cur_image_name)
    ax.axis('off')
    return rgb_image

    
def search(id_num, boxes, img, ax):
    found = False
    #rect = ''
    for box in boxes:
        if box[4]==id_num:
            
            # Create a Rectangle patch
            rect =patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                                     linewidth=2,edgecolor='r',facecolor='none')

            #makes sure object is in center
            found = center(rect, img) 
            #area_large_enough(rect)
                      
            # Add the patch to the Axes
            if found:
                ax.add_patch(rect)
                #break
                
    #draw the plot on the figure
    plt.draw()
    plt.pause(0.01)
    return found



def approach(images_path, cur_image_name, annotations, ax, id_num):
    #img = cv.imread(os.path.join(images_path,cur_image_name))
    found = False
    reach = False
    while not reach:
        
        
        next_image_name = command('w', cur_image_name, annotations)
        
        #try an extra step if not dead end
        if next_image_name != '':
            prev_image_name = cur_image_name
            cur_image_name = next_image_name
            img = plot_image(images_path, cur_image_name, annotations, ax)
            boxes = annotations[cur_image_name]['bounding_boxes']
            found = search(id_num, boxes, img, ax)
            #if target not found after taking extra step, go back
            if not found:
                cur_image_name = prev_image_name
                img = plot_image(images_path, cur_image_name, annotations, ax)
                boxes = annotations[cur_image_name]['bounding_boxes']
                found = search(id_num, boxes, img, ax)
                reach = True 
        #if dead end        
        else:
            img = plot_image(images_path, cur_image_name, annotations, ax)
            boxes = annotations[cur_image_name]['bounding_boxes']
            found = search(id_num, boxes, img, ax)
            #search complete!
            if found:
                reach = True
            #try again
            else:
                reach = False
                break
            
    return reach