# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:43:37 2020

@author: timot
"""

import numpy as np
import cv2 as cv
import os
import sys

'''root = 'C:/Capstone2020/ActiveVisionDataset/'

scene_name = input("Enter a scene name: ")
requested_instance = input("Enter name of target object: ")


scene_path = os.path.join(root, scene_name)

instance_found = vis.check_present_instance(scene_path, requested_instance)
if not instance_found:
    print("Instance requested is not in the selected scene.")'''
    

cap = cv.VideoCapture(0)
    
while (True):
    ret, img = cap.read()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray,5)
    edges = cv.Canny(gray,50,250)
    lines = cv.HoughLines(edges,1,np.pi/180,100)
    circles = cv.HoughCircles(blur,cv.HOUGH_GRADIENT,1,20,
                                  param1=200,param2=100,minRadius=0,maxRadius=0)
    #contours, hierarchy = cv.findContours(edges,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    contours, hierarchy = cv.findContours(edges,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
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
            cv.line(img,(x1,y1),(x2,y2),(0,0,255),3)'''
            
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line[0]
            cv.line(img,(x1,y1),(x2,y2),(0,0,255),5)
            
    '''if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(img,(i[0],i[1]),i[2],(0,255,0),5)
            # draw the center of the circle
            #cv.circle(rgb_image,(i[0],i[1]),2,(0,0,255),3)
            
    #cv.drawContours(img, contours, -1, (0, 255, 0), 3) '''
            
    # Searching through every region selected to  
    # find the required polygon. 
    for cnt in contours : 
            area = cv.contourArea(cnt) 
   
            # Shortlisting the regions based on there area. 
            if area > 200:  
                approx = cv.approxPolyDP(cnt,  
                                      0.009 * cv.arcLength(cnt, True), True) 
   
                # Checking if the no. of sides of the selected region is 7. 
                if(len(approx) == 4):  
                    cv.drawContours(img, [approx], 0, (255, 0, 0), 5)
            
    
    cv.imshow('frame',img)
    if cv.waitKey(1)==27:
        break
    
cap.release()
cv.destroyAllWindows