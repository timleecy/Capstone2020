# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 21:48:36 2020

@author: timot
"""

import os
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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

'''def plot_image(images_path, cur_image_name, ax):
    #load the current image and annotations 
    rgb_image = cv.imread(os.path.join(images_path,cur_image_name))
    plt.cla()
    ax.imshow(rgb_image)
    plt.title(cur_image_name)
    ax.axis('off')
    return rgb_image'''
    

def search(target_colour, target_shape, images_path, cur_image_name, ax):
    found = False
    
    black  = (0,179,0,255,0,36)
    white  = (0,179,0,26,230,255)
    red    = (0,8,127,255,127,255)
    orange = (9,20,127,255,127,255)
    yellow = (21,30,127,255,127,255)
    green  = (31,85,127,255,127,255)
    blue   = (86,125,127,255,127,255)
    purple = (126,140,127,255,127,255)
    pink   = (141,175,127,255,127,255)
    brown  = (9,20,127,255,51,153)
    colour = vars()[target_colour]
    lower_colour = np.array([colour[0], colour[2], colour[4]])
    upper_colour = np.array([colour[1], colour[3], colour[5]])
    
    img = cv.imread(os.path.join(images_path,cur_image_name))
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #edges = cv.Canny(gray,50,250,L2gradient=True)
    blur = cv.medianBlur(img,5)
    #blur = cv.GaussianBlur(img, (3, 3), 0)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv_blur = cv.cvtColor(blur,cv.COLOR_BGR2HSV)
    #lines = cv.HoughLinesP(edges,1,np.pi/180,10)
    
    
    mask = cv.inRange(hsv, lower_colour, upper_colour)
    mask_blur = cv.inRange(hsv_blur, lower_colour, upper_colour)
    kernel = np.ones((5,5), np.uint8)
    mask = cv.erode(mask, kernel)
    mask_blur = cv.erode(mask_blur, kernel)
    
    
    res = cv.bitwise_and(blur, blur, mask = mask_blur)
    gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 250)
    
        
    if target_shape == 'circle':
        circles = cv.HoughCircles(edges,cv.HOUGH_GRADIENT,1,20,
                                  param1=200,param2=100,minRadius=0,maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv.circle(img,(i[0],i[1]),i[2],(255,0,0),5)
                found = True
    
    else:
        contours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        #contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
        if target_shape == 'square' or 'rectangle':
            sides = 4
        elif target_shape == 'triangle':
            sides = 3
            
        for cnt in contours: 
            area = cv.contourArea(cnt) 
       
            # Shortlisting the regions based on there area. 
            if area > 5000:  
                approx = cv.approxPolyDP(cnt,  
                                      0.01 * cv.arcLength(cnt, True), True) 
       
                # Checking if the no. of sides of the selected region is 7. 
                if len(approx) == sides:   
                    cv.drawContours(img, [approx], -1, (255, 0, 0), 5)
                    found = True
                    
    
            
    #draw the plot on the figure
    plt.cla()
    ax.imshow(img)
    plt.title(cur_image_name)
    ax.axis('off')
    plt.pause(0.001)
    return found


def approach(images_path, cur_image_name, annotations, ax, target_colour, target_shape):
    found = False
    reach = False
    while not reach:
        
        next_image_name = command('w', cur_image_name, annotations)
        
        #try an extra step if not dead end
        if next_image_name != '':
            prev_image_name = cur_image_name
            cur_image_name = next_image_name
            found = search(target_colour, target_shape, images_path, cur_image_name, ax)
            #if target not found after taking extra step, go back
            if not found:
                cur_image_name = prev_image_name
                found = search(target_colour, target_shape, images_path, cur_image_name, ax)
                reach = True 
                
        #if dead end        
        else:
            found = search(target_colour, target_shape, images_path, cur_image_name, ax)
            #search complete!
            if found:
                reach = True
            #try again
            else:
                reach = False
                break
            
    return reach