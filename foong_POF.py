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
        rgb_image = cv2.imread(os.path.join(images_path,cur_image_name))
        boxes = annotations[cur_image_name]['bounding_boxes']
        
        
        
        #Setting up feature detection
        #Parameters for ShiTomasi Corder detection
        shito_params = dict( maxCorners = 100,
                             qualityLevel = 0.01,
                             minDistance = 7,
                             blockSize = 7 )

        #Parameters for Lucas Kanade Optical Flow
        lk_params = dict( winSize = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        
        #Create random colours
        #color = np.random.randint(0,255,(100,3))
        #print(color)
        
        #red colour list
        color = (255, 0 ,0)
        
        #Find corner of first frame
        #ret, prev_frame = cap
        cur_frame = 0
        prev_frame = rgb_image
        prev_frame_grey = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        #detecing standout corner of image
        p0 = cv2.goodFeaturesToTrack(prev_frame_grey, mask = None, **shito_params)

        
        #create mask for line drawing
        mask = np.zeros_like(prev_frame_grey)  #return array of zeroes similar to the image

        
        #plot the image and draw the boxes
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 960, 540)
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
            
            
        #Optical flow tracking when transition into the next frame    

        if True:
            print('checkpoint')
            #ret, cur_frame = cap
            cur_frame = rgb_image
            cur_frame_grey = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
            
            #Calculate Optical Flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame_grey, cur_frame_grey, p0, None, **lk_params)
            
            #Select good points
            good_cur = p1[st==1]
            good_prev = p0[st==1]
            
            #draw tracks
            for i,(cur,prev) in enumerate(zip(good_cur, good_prev)):
                a,b = cur.ravel()
                c,d = prev.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color, 2)
                cur_frame = cv2.circle(cur_frame, (a,b),5,color,-1)
                
        #print(cur_frame.shape)
        #print(mask.shape)

        #img = cv2.addWeighted(cur_frame_list[0:1], 0.5, mask, 0.5, 0)
        cv2.namedWindow('optical tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('optical tracking', 640, 360)
        cv2.imshow('optical tracking', mask)
        
        #img = cv2.addWeighted(cur_frame_list[0:1], 0.5, mask, 0.5, 0)
        cv2.namedWindow('point detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('point detection', 640, 360)
        cv2.imshow('point detection', cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY))
                        
        #update previous frame and previous points
        prev_frame_grey = cur_frame_grey.copy()
        p0 = good_cur.reshape(-1,1,2)
    
        



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