# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 11:11:27 2020

@author: timot
"""

import cv2 as cv
import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt

img = cv.imread('C:/Capstone2020/black_tv_screen.jpg', cv.IMREAD_GRAYSCALE)

root = 'C:/Capstone2020/ActiveVisionDataset/'
scene_name = 'Home_005_1'
scene_path = os.path.join(root,scene_name)

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

# Features
sift = cv.xfeatures2d.SIFT_create()
kp_img, desc_img = sift.detectAndCompute(img, None)
# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv.FlannBasedMatcher(index_params, search_params)

key = 0
while True:
    #frame = cv.imread('C:/Capstone2020/ActiveVisionDataset/Home_005_1/jpg_rgb/000510000340101.jpg')
    frame = cv.imread(os.path.join(images_path,cur_image_name))
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp_frame, desc_frame = sift.detectAndCompute(gray_frame, None)
    matches = flann.knnMatch(desc_img, desc_frame, k=2)
    
    good_points = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_points.append(m)
            
    #img3 = cv.drawMatches(img, kp_img, frame, kp_frame, good_points, frame)
    
    if len(good_points)>10:
        query_pts = np.float32([kp_img[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
    
        # Perspective transform
        h, w = img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, matrix)
        
        homography = cv.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        cv.resizeWindow('image', 960, 540)
        cv.imshow('image', homography)
        '''plt.cla()
        ax.imshow(homography)
        plt.title(cur_image_name)
        ax.axis('off')
        plt.pause(0.001)'''
        key = cv.waitKey(1)
        
    else:
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        cv.resizeWindow('image', 960, 540)
        cv.imshow('image', frame)
        '''plt.cla()
        ax.imshow(homography)
        plt.title(cur_image_name)
        ax.axis('off')
        plt.pause(0.001)'''
        key = cv.waitKey(1)
        
            
    #move_command = input('Enter command: ')
    
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
            
            
    
    '''#get the next image name to display based on the 
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
              "'h' - print this help menu"))'''
    
    
    #if the user inputted move is valid (there is an image there) 
    #then update the image to display. If the move was not valid, 
    #the current image will be displayed again
    if next_image_name != '':
        cur_image_name = next_image_name
        


