# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 15:07:37 2020

@author: timot
"""


#import init as init #has file paths
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import misc
import scipy.io as sio
import cv2 as cv
import numpy as np

def check_present_instance(scene_path, requested_instance):
    present_instance_path = os.path.join(scene_path,'present_instance_names.txt')
    present_instances = open(present_instance_path,'r')
    present_instances = present_instances.readlines()
    
    present_instances_list = []
    for line in present_instances:
        present_instances_list.append(line.strip())
        
    present=False;
    for instance in present_instances_list:
        if instance == requested_instance:
            present=True;
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
        
def nothing():
    pass
    
                       

def vis_boxes_and_move(root, scene_path, requested_instance):
    """ Visualizes bounding boxes and images in the scene.

    Allows user to navigate the scene via the movement 
    pointers using the keyboard


    ARGUMENTS:
        scene_path: the string full path of the scene to view
            Ex) vis_camera_pos_dirs('/path/to/data/Home_01_1')

    """
    
    name_to_id_dict = get_instance_name_to_id_dict(root)
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
    while (move_command != 'q'):

        #load the current image and annotations 
        rgb_image = cv.imread(os.path.join(images_path,cur_image_name))
        boxes = annotations[cur_image_name]['bounding_boxes']
        
        #hough lines & circles
        gray = cv.cvtColor(rgb_image,cv.COLOR_BGR2GRAY)
        hsv = cv.cvtColor(rgb_image,cv.COLOR_BGR2HSV)
        edges = cv.Canny(gray,50,250,L2gradient=True)
        blur = cv.medianBlur(gray,5)
        lines = cv.HoughLinesP(edges,1,np.pi/180,10)
        circles = cv.HoughCircles(blur,cv.HOUGH_GRADIENT,1,20,
                                  param1=200,param2=100,minRadius=0,maxRadius=0)
        contours, hierarchy = cv.findContours(edges,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        #contours, hierarchy = cv.findContours(edges,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        
        '''if lines is not None:
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv.line(rgb_image,(x1,y1),(x2,y2),(0,0,255),5)'''
                
        '''if lines is not None:
            for line in lines:
                x1,y1,x2,y2=line[0]
                cv.line(rgb_image,(x1,y1),(x2,y2),(0,0,255),5)'''
            
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv.circle(rgb_image,(i[0],i[1]),i[2],(0,255,0),5)
                # draw the center of the circle
                #cv.circle(rgb_image,(i[0],i[1]),2,(0,0,255),3)
                
        #cv.drawContours(rgb_image, contours, -1, (0, 255, 0), 3) 
        
                
        '''for cnt in contours : 
            area = cv.contourArea(cnt) 
   
            # Shortlisting the regions based on there area. 
            if area > 200:  
                approx = cv.approxPolyDP(cnt,  
                                      0.009 * cv.arcLength(cnt, True), True) 
   
                # Checking if the no. of sides of the selected region is 7. 
                if(len(approx) == 4):  
                    cv.drawContours(rgb_image, [approx], -1, (255, 0, 0), 5)'''
        
                                 
     
        #plot the image and draw the boxes
        plt.cla()
        ax.imshow(rgb_image)
        plt.title(cur_image_name)
        plt.pause(.001)

        #get input from user 
        move_command = input('Enter command: ')
    


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


        #if the user inputted move is valid (there is an image there) 
        #then update the image to display. If the move was not valid, 
        #the current image will be displayed again
        if next_image_name != '':
            cur_image_name = next_image_name








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
    





