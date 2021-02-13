# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:59:32 2020

@author: timot
"""


import cv2 as cv
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
#images = glob.glob('*.jpg')
calib_path = 'C:/Capstone2020/calibration/'
calib_names = os.listdir(calib_path)

for fname in calib_names:
    calib_img = cv.imread(os.path.join(calib_path, fname))
    calib_gray = cv.cvtColor(calib_img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(calib_gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(calib_gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(calib_img, (7,6), corners2, ret)
        cv.imshow('calib', calib_img)
        cv.waitKey(500)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, calib_gray.shape[::-1], None, None)


root = 'C:/Capstone2020/ActiveVisionDataset/'
scene_name = sys.argv[1]
scene_path = os.path.join(root,scene_name)

 
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




while True:

    img = cv.imread(os.path.join(images_path,cur_image_name))
    h, w = img.shape[:2]
    newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    print(dst.shape)
    cv.imshow('calibresult.png',dst)
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



    
  
        
  

