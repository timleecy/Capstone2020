# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 21:59:11 2020

@author: timot
"""


import cv2 as cv
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import project_modules as pm
import msvcrt
import skimage
from skimage import transform

black  = (0,179,0,255,0,64)
white  = (0,179,0,64,192,255)
red    = (0,8,127,255,64,255)
orange = (9,20,127,255,127,255) #brown included
yellow = (21,30,127,255,64,255)
green  = (31,85,127,255,64,255)
blue   = (86,125,127,255,64,255)
purple = (126,140,127,255,64,255)
pink   = (141,175,127,255,64,255)
brown  = (9,20,127,255,64,192)
gold = (0,179,128,255,51,128)
test = (0,179,0,255,0,255)

target_colour = 'yellow'
target_shape = 'circle'

colour = vars()[target_colour]
lower_colour = np.array([colour[0], colour[2], colour[4]])
upper_colour = np.array([colour[1], colour[3], colour[5]])

img = cv.imread('C:/Capstone2020/shapes.jpg')
blur = cv.medianBlur(img,5)
#blur = cv.GaussianBlur(img, (3, 3), 0)
hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
#hsv_blur = cv.cvtColor(blur,cv.COLOR_BGR2HSV)
    
mask = cv.inRange(hsv, lower_colour, upper_colour)

'''mask_blur = cv.inRange(hsv_blur, lower_colour, upper_colour)
kernel = np.ones((5,5), np.uint8)
mask = cv.erode(mask, kernel)
mask_blur = cv.erode(mask_blur, kernel)
  
res = cv.bitwise_and(blur, blur, mask=mask_blur)
gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)'''


'''if target_shape == 'circle':
    #res = cv.bitwise_and(blur, blur, mask=mask)
    #gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(mask, 50, 250)
    
    circles = cv.HoughCircles(edges,cv.HOUGH_GRADIENT,1,500)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(img,(i[0],i[1]),i[2],(255,0,0),5)
            #found = True
    
else:'''
contours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
#contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
for cnt in contours: 
    area = cv.contourArea(cnt) 
   
    # Shortlisting the regions based on there area. 
    if area > 50:  
        approx = cv.approxPolyDP(cnt,  
                              0.05 * cv.arcLength(cnt, True), True) 
        
        if target_shape == 'square' and len(approx) == 4:
            x,y,w,h = cv.boundingRect(approx)
            aspectRatio = float(w)/h
            if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                cv.drawContours(img, [approx], 0, (0, 0, 255), 2)
                
        elif target_shape == 'rectangle' and len(approx) == 4:
            x,y,w,h = cv.boundingRect(approx)
            aspectRatio = float(w)/h
            if aspectRatio <= 0.95 or aspectRatio >= 1.05:
                cv.drawContours(img, [approx], 0, (0, 0, 255), 2)
                
        elif target_shape == 'triangle' and len(approx) == 3:
            cv.drawContours(img, [approx], 0, (0, 0, 255), 2)
            
        elif target_shape == 'circle' and len(approx) >= 10: 
            cv.drawContours(img, [approx], 0, (0, 0, 255), 2)
                
 
cv.imshow('image',img)
cv.imshow('mask', mask)
#cv.imshow('edges', edges)
key = cv.waitKey(-1)      
if key == 113:
    cv.destroyAllWindows()