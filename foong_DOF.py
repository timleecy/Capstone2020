# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:17:52 2020

@author: timot
"""


#import init as init #has file paths
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import misc
import scipy.io as sio
import cv2
import numpy as np
import argparse


def vis_boxes_and_move(scene_path):
    """ Visualizes bounding boxes and images in the scene.
    Allows user to navigate the scene via the movement 
    pointers using the keyboard
    ARGUMENTS:
        scene_path: the string full path of the scene to view
            Ex) vis_camera_pos_dirs('/path/to/data/Home_01_1')
    """


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
    #fig,ax = plt.subplots(1)
    
    key = 0
    while True:

        #load the current image and annotations 
        read_image = cv2.imread(os.path.join(images_path,cur_image_name))
        scale_percent = 20 #percent of orginal size
        width = int(read_image.shape[1]*scale_percent/100)
        height = int(read_image.shape[0]*scale_percent/100)
        dimension = (width, height)
        rgb_image = cv2.resize(read_image,dimension,interpolation = cv2.INTER_AREA)
        boxes = annotations[cur_image_name]['bounding_boxes']
  
    
        #Start of dense optical flow
        prev_frame = rgb_image
        prev_frame_grey = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(prev_frame)
        hsv[...,1] = 255
        

        #plot the image and draw the boxes
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 480, 270)
        cv2.imshow('image', rgb_image)

        key = cv2.waitKey(-1)
        
        '''
        plt.cla()
        ax.imshow(rgb_image)
        plt.title(cur_image_name)
        
        for box in boxes:
            # Create a Rectangle patch
            rect =patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                                     linewidth=2,edgecolor='r',facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
        #draw the plot on the figure
        plt.draw()
        plt.pause(.001)
        '''
        
        #---------USER COMMAND FOR DEBUG-----------#

        #get input from user 
        #move_command = input('Enter command: ')


        #get the next image name to display based on the 
        #user input, and the annotation.
        if key == 119:
            next_image_name = annotations[cur_image_name]['forward']
        elif key == 97:
            next_image_name = annotations[cur_image_name]['rotate_ccw']
        elif key == 115:
            next_image_name = annotations[cur_image_name]['backward']
        elif key == 100:
            next_image_name = annotations[cur_image_name]['rotate_cw']
        elif key == 101:
            next_image_name = annotations[cur_image_name]['left']
        elif key == 114:
            next_image_name = annotations[cur_image_name]['right']
        elif key == 104:
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
        elif key == 113:
            cv2.destroyAllWindows
            break
        
       #if the user inputted move is valid (there is an image there) 
        #then update the image to display. If the move was not valid, 
        #the current image will be displayed again
        if next_image_name != '':
            cur_image_name = next_image_name
        
        if True:
            print('checkpoint')
            cur_frame = rgb_image
            cur_frame_grey = cv2.cvtColor(cur_frame,cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(prev_frame_grey, cur_frame_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            
            dense_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            
            cv2.namedWindow('Dense Optical Flow', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Dense Optical Flow', 480, 270)
            cv2.imshow('Dense Optical Flow', dense_flow)
            
            prev_frame = cur_frame



def vis_camera_pos_dirs(scene_path, plot_directions=True, scale_positions=True):
    """ Visualizes camera positions and directions in the scene.
    ARGUMENTS:
        scene_path: the string full path of the scene to view
            Ex) vis_camera_pos_dirs('/path/to/data/Home_01_1')
    KEYWORD ARGUMENTS:
        plot_directions: bool, whether or not to plot camera directions
                         defaults to True
            Ex) vis_camera_pos_dirs('Home_01_1', plot_directions=False)
        scale_positions: bool, whether or not to scale camera positions 
                         to be in millimeters. Defaults to True
            Ex) vis_camera_pos_dirs('Home_01_1', scale_positions=False)
    """

    #TODO - make faster - organize all positions/directions, then plot

    #set up scene specfic paths
    images_path = os.path.join(scene_path,'jpg_rgb')
    image_structs_path = os.path.join(scene_path,'image_structs.mat')

    #load data.
    #the transition from matlab to python is not pretty
    image_structs = sio.loadmat(image_structs_path)
    scale = image_structs['scale'][0][0]
    image_structs = image_structs['image_structs']
    image_structs = image_structs[0]

    #make plot
    fig,ax = plt.subplots(1)

    for camera in image_structs:
        #get 3D camera position in the reconstruction
        #coordinate frame. The scale is arbitrary
        world_pos = camera[3] 
        if scale_positions:
            world_pos *= scale

        #get 3D vector that indicates camera viewing direction
        #Add the world_pos to translate the vector from the origin
        #to the camera location.
        camera[4] /= 2;#to make plot look nicer
        if scale_positions:
            direction = world_pos + camera[4]*scale
        else:
            direction = world_pos + camera[4]
            
        #plot only 2D, as all camera heights are the same

        #draw the position
        plt.plot(world_pos[0], world_pos[2],'ro')    
        #draw the direction if user sets option 
        if plot_directions:
            plt.plot([world_pos[0], direction[0]], 
                             [world_pos[2], direction[2]], 'b-')    


    #for camera in image_structs 
    plt.axis('equal')
    plt.show()    