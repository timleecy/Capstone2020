# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:40:49 2020

@author: timot
"""

import cv2 as cv
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


black  = (0,179,0,255,0,40)
white  = (0,179,0,64,192,255)
red1   = (0,8,64,255,64,255)
red2   = (141,179,64,255,64,255) #includes pink
orange = (9,20,64,255,64,255) #brown may overlap
yellow = (21,30,64,255,64,255)
green  = (31,85,64,255,64,255)
blue   = (86,125,64,255,64,255)
purple = (126,140,64,255,64,255)
test = (0,179,0,255,0,255)


root = 'C:/Capstone2020/ActiveVisionDataset/'
scene_name = sys.argv[1]
scene_path = os.path.join(root,scene_name)
target_colour = sys.argv[2]
target_shape = sys.argv[3]

if target_colour == 'red':
    lower_colour_1 = np.array([red1[0], red1[2], red1[4]])
    upper_colour_1 = np.array([red1[1], red1[3], red1[5]])
    lower_colour_2 = np.array([red2[0], red2[2], red2[4]])
    upper_colour_2 = np.array([red2[1], red2[3], red2[5]])
else:
    colour = vars()[target_colour]
    lower_colour = np.array([colour[0], colour[2], colour[4]])
    upper_colour = np.array([colour[1], colour[3], colour[5]])
    
 
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

#set up tracking plot
x_axis=0
y_axis=0
dx=0
dy=0
theta=-30

fig,ax = plt.subplots()     
objects = np.array([])
found = False
while True:

    img = cv.imread(os.path.join(images_path,cur_image_name))
    
    if target_shape == 'circle':
        blur = cv.medianBlur(img, 5)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        
        if target_colour == 'red':
            mask1 = cv.inRange(hsv, lower_colour_1, upper_colour_1)
            mask2 = cv.inRange(hsv, lower_colour_2, upper_colour_2)
            mask = cv.bitwise_or(mask1, mask2)
        else:
            mask = cv.inRange(hsv, lower_colour, upper_colour)      
            
        res = cv.bitwise_and(blur, blur, mask=mask)
        gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 300)   
        circles = cv.HoughCircles(edges,cv.HOUGH_GRADIENT,1,500)
                              #param1=200,param2=100,minRadius=0,maxRadius=0)
        cv.imshow('mask', res)
        cv.imshow('gray', gray)
        cv.imshow('edges', edges)
        if circles is not None:
            circles = np.uint16(np.around(circles))        
            for i in circles[0,:]:      
                area = np.pi*((i[2])**2)
                
                if area>3000:
                    cv.circle(img,(i[0],i[1]),i[2],(0,0,255),5)
                    found = True
                
                '''if found:
                    cnt_img = np.zeros_like(mask)
                    cnt_img = cv.circle(cnt_img,(i[0],i[1]),i[2],255,5)
                    pixelpoints = np.transpose(np.nonzero(cnt_img))                   
                    depth = cv.imread(os.path.join(depth_path,cur_depth_name), cv.IMREAD_ANYDEPTH)
                    #depth = np.float32(depth)
                    #normalised_depth = np.zeros(depth.shape)
                    #normalised_depth = cv.bilateralFilter(depth, 3,75,75)
                    #depth = normalised_depth
                    res_row = (2*depth*np.tan((70*np.pi/180)/2))/depth.shape[0]
                    res_col = (2*depth*np.tan((60*np.pi/180)/2))/depth.shape[1]                    
                    real_area = 0
                    
                    nonzero_row = res_row[np.nonzero(res_row)]
                    nonzero_col = res_col[np.nonzero(res_col)]
                    row_mean = np.mean(nonzero_row)
                    col_mean = np.mean(nonzero_col)
                    
                    for pixel in pixelpoints:            
                        
                        
                        if res_row[pixel[0]][pixel[1]]==0:
                            row_real = row_mean
                        else:
                            row_real = res_row[pixel[0]][pixel[1]]
                            
                        if res_col[pixel[0]][pixel[1]]==0:
                            col_real = col_mean
                        else:
                            col_real = res_col[pixel[0]][pixel[1]]
                        
                        real_area += row_real*col_real
    
                    print(real_area)
                    
                    objects = np.append(objects, cur_image_name)
                    objects = np.append(objects, real_area) #change area to actual size'''

                
    else:
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hsv = cv.medianBlur(hsv, 7) #stick to 7
        
        if target_colour == 'red':
            mask1 = cv.inRange(hsv, lower_colour_1, upper_colour_1)
            mask2 = cv.inRange(hsv, lower_colour_2, upper_colour_2)
            mask = cv.bitwise_or(mask1, mask2)
        else:
            mask = cv.inRange(hsv, lower_colour, upper_colour)   
        
        #kernel = np.ones((7,7), np.uint8) #stick to 7
        #mask = cv.erode(mask, kernel) #must have erode
        #edges = cv.Canny(mask, 50, 250)
        
       
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        #cv.drawContours(img, contours, -1, (0,255,0), 3)
        #cv.imshow('mask', img)
        
        biggest_cnt_area = 0
        biggest_cnt = 0
        for cnt in contours: 
            
            area = cv.contourArea(cnt) 
            
            # Shortlisting the regions based on there area. 
            if area > 2000 and area<500000:  #stick to 3000 for now
                approx = cv.approxPolyDP(cnt,  
                                     0.08 * cv.arcLength(cnt, True), True) #dont change 0.05
                approx = np.squeeze(approx)
 
  
                if target_shape == 'square' and len(approx) == 4:
                    x,y,w,h = cv.boundingRect(approx)
                    aspectRatio = float(w/h)
                    if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                        if cv.isContourConvex(approx):
                            cv.drawContours(img, [approx], 0, (0, 0, 255), 5)
                            found = True
                        
                elif target_shape == 'rectangle' and len(approx) == 4:
                    x,y,w,h = cv.boundingRect(approx)
                    aspectRatio = float(w/h)
                    if aspectRatio <= 0.95 or aspectRatio >= 1.05:
                        if cv.isContourConvex(approx):
                            
                            cy = y + h/2
                            if cy > 1080*0.1 and cy < 1080*0.9:
                                #cv.drawContours(img, [approx], 0, (0, 0, 255), 5)                    
                                found = True
                                if area > biggest_cnt_area:
                                    biggest_cnt_area = area
                                    biggest_cnt = approx
                              
                elif target_shape == 'triangle' and len(approx) == 3:
                    cv.drawContours(img, [approx], 0, (0, 0, 255), 5)
                    found = True
                    
                '''if found:
                    cv.drawContours(img, [biggest_cnt], 0, (0, 0, 255), 5) '''
                    
                        
                '''if found:
                    cnt_img = np.zeros_like(mask)
                    cnt_img = cv.fillPoly(cnt_img, [approx], 255)
                    #cv.imshow('cimg', cnt_img)
                    pixelpoints = np.transpose(np.nonzero(cnt_img))
                    depth = cv.imread(os.path.join(depth_path,cur_depth_name), cv.IMREAD_ANYDEPTH)
                    depth = depth*1e-3
                    #depth = np.float32(depth)
                    #normalised_depth = np.zeros(depth.shape)
                    #normalised_depth = cv.bilateralFilter(depth, 3,75,75)
                    #depth = normalised_depth

                    cnt_depth = []
                    for pixel in pixelpoints:
                        cnt_depth.append(depth[pixel[0]][pixel[1]])        
                    cnt_depth = np.array(cnt_depth)
                    nonzero_depth = cnt_depth[np.nonzero(cnt_depth)]
                    depth_mean = np.mean(nonzero_depth)
                    
                    real_area = 0
                    for depth_pixel in cnt_depth:
                        
                        if depth_pixel == 0:
                            res_row = (2*depth_mean*np.tan((70*np.pi/180)/2))/depth.shape[1]
                            res_col = (2*depth_mean*np.tan((60*np.pi/180)/2))/depth.shape[0]
                        else:
                            res_row = (2*depth_pixel*np.tan((70*np.pi/180)/2))/depth.shape[1]
                            res_col = (2*depth_pixel*np.tan((60*np.pi/180)/2))/depth.shape[0]
                        
                            
                        real_area += res_row*res_col
                      
                        
                    print(real_area)
                    objects = np.append(objects, cur_image_name)
                    objects = np.append(objects, real_area) #change area to actual size'''
    
    
    plt.xlim(-20,20)
    plt.ylim(-20,20) 
    plt.arrow(x_axis, y_axis, dx, dy)
    if found:
        #d=depth_mean*1e-3
        #object_pos_x = d*np.tan(obj_angle*np.pi/180)
        #object_pos_y = d*np.tan((45-theta)*np.pi/180)
        #object_pos_x_rotated = object_pos_x*np.cos(theta*np.pi/180) - object_pos_y*np.sin(theta*np.pi/180)
        #object_pos_y_rotated = object_pos_x*np.sin(theta*np.pi/180) + object_pos_y*np.cos(theta*np.pi/180)
        #plt.plot(x_axis+dx+object_pos_x_rotated, y_axis+dy+object_pos_y_rotated, 'b+')   
        plt.plot(x_axis+dx, y_axis+dy, 'b+')   
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

'''objects = np.reshape(objects, (-1,2))
objects = np.unique(objects, axis = 0)
print(objects)'''


    
  
        
  

