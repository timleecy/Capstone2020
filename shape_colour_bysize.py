# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 23:02:57 2020

@author: timot
"""

import cv2 as cv
import numpy as np
import os
import sys
import json
import project_modules as pm

black  = (0,179,0,255,0,32)
white  = (0,179,0,64,192,255)
red    = (0,8,127,255,64,255)
orange = (9,20,127,255,127,255) #brown may overlap
yellow = (21,30,127,255,64,255)
green  = (31,85,127,255,64,255)
blue   = (86,125,127,255,64,255)
purple = (126,140,127,255,64,255)
pink   = (141,175,127,255,64,255)
brown  = (9,20,127,255,64,192)
gold = (0,179,128,255,51,128)
test = (0,179,0,255,0,255)


root = 'C:/Capstone2020/ActiveVisionDataset/'
scene_name = sys.argv[1]
scene_path = os.path.join(root,scene_name)
target_colour = sys.argv[2]
target_shape = sys.argv[3]

colour = vars()[target_colour]
lower_colour = np.array([colour[0], colour[2], colour[4]])
upper_colour = np.array([colour[1], colour[3], colour[5]])
'''
print('Press \'Esc\' once you are done choosing colour range')
def nothing(x):
    pass
cv.namedWindow('Trackbars')
cv.createTrackbar('Lower Hue', 'Trackbars', 0, 179, nothing)
cv.createTrackbar('Upper Hue', 'Trackbars', 0, 179, nothing)
cv.createTrackbar('Lower Sat', 'Trackbars', 0, 255, nothing)
cv.createTrackbar('Upper Sat', 'Trackbars', 0, 255, nothing)
cv.createTrackbar('Lower Val', 'Trackbars', 0, 255, nothing)
cv.createTrackbar('Upper Val', 'Trackbars', 0, 255, nothing)
key = cv.waitKey(-1)

if key == 27:
    lower_hue = cv.getTrackbarPos('Lower Hue', 'Trackbars')
    upper_hue = cv.getTrackbarPos('Upper Hue', 'Trackbars')
    lower_sat = cv.getTrackbarPos('Lower Sat', 'Trackbars')
    upper_sat = cv.getTrackbarPos('Upper Sat', 'Trackbars')
    lower_val = cv.getTrackbarPos('Lower Val', 'Trackbars')
    upper_val = cv.getTrackbarPos('Upper Val', 'Trackbars')
    lower_colour = np.array([lower_hue, lower_sat, lower_val])
    upper_colour = np.array([upper_hue, upper_sat, upper_val])
    
    cv.destroyAllWindows()'''
    
 
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


while True:

    img = cv.imread(os.path.join(images_path,cur_image_name))

    if target_shape == 'circle':
        blur = cv.medianBlur(img, 3)
        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_colour, upper_colour)      
        res = cv.bitwise_and(blur, blur, mask=mask)
        gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 250)   
        circles = cv.HoughCircles(edges,cv.HOUGH_GRADIENT,1,20,
                              param1=200,param2=100,minRadius=0,maxRadius=0)
        cv.imshow('mask', mask)
        
        if circles is not None:
            #circles = np.uint16(np.around(circles))
            
            biggest_circle_area = 0
            biggest_circle = 0
            for circ in circles[0,:]:
                area = np.pi*((circ[2])**2)
                
                if area > biggest_circle_area:
                    biggest_circle_area = area
                    biggest_circle = np.uint16(np.around(circ))
                    
            if biggest_circle_area > 10000:
                cv.circle(img,(biggest_circle[0],biggest_circle[1]),biggest_circle[2],(0,0,255),5)
            
            #for i in circles[0,:]:      
                #cv.circle(img,(i[0],i[1]),i[2],(0,0,255),5)
                
    else:
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        blur = cv.medianBlur(hsv, 7)
        mask = cv.inRange(blur, lower_colour, upper_colour)
        kernel = np.ones((5,5), np.uint8)
        mask = cv.erode(mask, kernel)
        edges = cv.Canny(mask, 50, 250)
     
        cv.imshow('mask', mask)

        contours, hierarchy = cv.findContours(edges ,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
        biggest_cnt_area = 0
        biggest_cnt = 0    
        for cnt in contours: 
            area = cv.contourArea(cnt) 
            
            if area > biggest_cnt_area:
                biggest_cnt_area = area
                biggest_cnt = cnt
       
        # Shortlisting the regions based on there area. 
        if biggest_cnt_area > 7000:  
            approx = cv.approxPolyDP(biggest_cnt,  
                                  0.05 * cv.arcLength(biggest_cnt, True), True) #dont change 0.05
            
            
            if target_shape == 'square' and len(approx) == 4:
                x,y,w,h = cv.boundingRect(approx)
                aspectRatio = float(w/h)
                if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                    cv.drawContours(img, [approx], 0, (0, 0, 255), 5)
                    
            elif target_shape == 'rectangle' and len(approx) == 4:
                x,y,w,h = cv.boundingRect(approx)
                aspectRatio = float(w/h)
                if aspectRatio <= 0.95 or aspectRatio >= 1.05:
                    cv.drawContours(img, [approx], 0, (0, 0, 255), 5)
                    
            elif target_shape == 'triangle' and len(approx) == 3:
                cv.drawContours(img, [approx], 0, (0, 0, 255), 5)
                
    
               
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', 960, 540)
    cv.imshow('image',img)
    key = cv.waitKey(-1)
    
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
    elif key==104:
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
    elif key==113:
        cv.destroyAllWindows
        break
 
    
    if next_image_name != '':
        cur_image_name = next_image_name
        


   
    
  
        
  

