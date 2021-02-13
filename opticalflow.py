# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 22:49:47 2020

@author: timot
"""


import numpy as np
import cv2 as cv
import argparse
import sys
import os
import json
import matplotlib.pyplot as plt
from project_modules2 import *
import math

''''parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()
cap = cv.VideoCapture(args.image)'''



#set path and user inputs
root = 'C:/Capstone2020/ActiveVisionDataset/'
scene_name = sys.argv[1]
scene_path = os.path.join(root,scene_name)
target_colour = sys.argv[2]
target_shape = sys.argv[3]
    
#set up scene specific paths
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
x_co = 0
y_co = 0
dx = 0
dy = 0
theta = -30

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.01,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
old_frame = cv.imread(os.path.join(images_path,cur_image_name))
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

#start main loop
fig,ax = plt.subplots()     
object_area = []
found = False
done = False
times_detected = 0
waited = False
cur_obj_area = 0
prev_obj_area = None
found_obj_coordinates = []
while True:
   
    frame = cv.imread(os.path.join(images_path,cur_image_name))
    img = frame
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    #cv.imshow('frame',img)
    #key = cv.waitKey(-1)
    '''k = cv.waitKey(30) & 0xff
    if k == 27:
        break'''
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    
    key, x_co, y_co = display(x_co, y_co, dx, dy, found, fig, img)
  
    if key == 113:
        cv.destroyAllWindows
        break
    else:
        next_image_name = command(key, annotations, cur_image_name)
    
    
    cur_image_name, cur_depth_name, dx, dy, theta = update_pos_img(cur_image_name, next_image_name, key, theta)
   
    
        
        




