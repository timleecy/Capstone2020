'''
VISION-BASED NAVIGATION AND OBJECT DETECTION AT LOW FRAME RATES
HELPER FUNCTIONS
Prepared by:
Timothy Lee (793942) leet4@student.unimelb.edu.au
Chuah Yee Hean (794709) chuahy1@student.unimelb.edu.au
Foong Shen Hui (816355) foongs@student.unimelb.edu.au

Date: 30/10/2020
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def get_colour_range(target_colour):
        
    #colour codes
    black  = (0,179,0,255,0,36)
    white  = (0,179,0,64,180,255)
    red1   = (0,8,64,255,64,255)
    red2   = (141,179,64,255,64,255) 
    orange = (9,20,64,255,64,255) 
    yellow = (21,30,64,255,64,255)
    green  = (31,85,64,255,64,255)
    blue   = (86,125,64,255,64,255)
    purple = (126,140,64,255,64,255)
    
    #set colour range
    if target_colour == 'red':
        lower_colour_1 = np.array([red1[0], red1[2], red1[4]])
        upper_colour_1 = np.array([red1[1], red1[3], red1[5]])
        lower_colour_2 = np.array([red2[0], red2[2], red2[4]])
        upper_colour_2 = np.array([red2[1], red2[3], red2[5]])
    else:
        colour = vars()[target_colour]
        lower_colour_1 = np.array([colour[0], colour[2], colour[4]])
        upper_colour_1 = np.array([colour[1], colour[3], colour[5]])
        lower_colour_2 = None
        upper_colour_2 = None
        
    return lower_colour_1, upper_colour_1, lower_colour_2, upper_colour_2


def search_circle(img, target_colour):
    
    found = False
    found_object = None
    lower_colour_1, upper_colour_1, lower_colour_2, upper_colour_2 = get_colour_range(target_colour)
    
    blur = cv.medianBlur(img, 9)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 
    if target_colour == 'red':
        mask1 = cv.inRange(hsv, lower_colour_1, upper_colour_1)
        mask2 = cv.inRange(hsv, lower_colour_2, upper_colour_2)
        mask = cv.bitwise_or(mask1, mask2)
    else:
        mask = cv.inRange(hsv, lower_colour_1, upper_colour_1)      
        
    edges = cv.Canny(mask, 50, 300)   
    circles = cv.HoughCircles(edges,cv.HOUGH_GRADIENT,1,500)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))        
        biggest_circle = 0
        biggest_circle_area = 0
        for i in circles[0,:]:      
            area = np.pi*((i[2])**2)
            
            if area>3000:
                img = cv.circle(img,(i[0],i[1]),i[2],(0,0,255),5)
                found = True
                if area > biggest_circle_area:
                    biggest_circle_area = area
                    biggest_circle = i

    if found:
        img = cv.circle(img,(i[0],i[1]),i[2],(0,0,255),5)
        found_object = biggest_circle
        
    return found, img, found_object


def search_polygon(img, target_shape, target_colour):
    
    found = False
    found_object = None
    lower_colour_1, upper_colour_1, lower_colour_2, upper_colour_2 = get_colour_range(target_colour)
        
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = cv.medianBlur(hsv, 9) 
    
    if target_colour == 'red':
        mask1 = cv.inRange(hsv, lower_colour_1, upper_colour_1)
        mask2 = cv.inRange(hsv, lower_colour_2, upper_colour_2)
        mask = cv.bitwise_or(mask1, mask2)
    else:
        mask = cv.inRange(hsv, lower_colour_1, upper_colour_1)   

    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    biggest_cnt_area = 0
    biggest_cnt = 0
    for cnt in contours: 
        area = cv.contourArea(cnt) 

        if area>30000 and area<500000:
    
            approx = cv.approxPolyDP(cnt, 0.07 * cv.arcLength(cnt, True), True) 
            approx = np.squeeze(approx)

            if target_shape == 'square' and len(approx) == 4:
                
                x,y,w,h = cv.boundingRect(approx)
                aspectRatio = float(w/h)
                if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                    if cv.isContourConvex(approx):
                        if isParallelogram(approx):
                            found = True
                            if area > biggest_cnt_area:
                                biggest_cnt_area = area
                                biggest_cnt = approx
                                               
            elif target_shape == 'rectangle' and len(approx) == 4: 
                x,y,w,h = cv.boundingRect(approx)
                aspectRatio = float(w/h)
                if aspectRatio <= 0.95 or aspectRatio >= 1.05:
                    if cv.isContourConvex(approx):     
                        if isParallelogram(approx):    
                            found = True
                            if area > biggest_cnt_area:
                                biggest_cnt_area = area
                                biggest_cnt = approx
                                                       
            elif target_shape == 'triangle' and len(approx) == 3:
                found = True
                if area > biggest_cnt_area:
                    biggest_cnt_area = area
                    biggest_cnt = approx
                                
    if found:
        img = cv.drawContours(img, [biggest_cnt], 0, (0, 0, 255), 5)
        found_object = np.squeeze(biggest_cnt)
        
    return found, img, found_object


def command_disp(key, annotations, images_path, cur_image_name, img, x_co, y_co, theta, fig, found):
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
        
    if next_image_name != '':
        cur_image_name = next_image_name
        distance = 0.3
        
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
        cur_image_name = cur_image_name
        dx = 0
        dy = 0
    
    
    plt.xlim(-10,10)
    plt.ylim(-10,10) 
    plt.arrow(x_co, y_co, dx, dy)
    if found:
        plt.plot(x_co+dx, y_co+dy, 'b+')   
    x_co += dx
    y_co += dy
    
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
    key = cv.waitKey(100)
    
    next_img = cv.imread(os.path.join(images_path, next_image_name))
    
    return next_image_name, next_img, x_co, y_co, theta, fig


def obs_detect(depth_image, threshold = 500):
    left_flag = 0
    mid_flag = 0
    right_flag = 0
    copy = depth_image.copy()
    second_smallest = sorted(list(set(copy.flatten().tolist())))[2]
    result = np.where(depth_image == second_smallest)
    coord_list = list(zip(result[0],result[1]))
    
    res = depth_image.shape[1]
    for row,col in coord_list:
        if depth_image[row][col] < threshold:
            if col > res*0.1 and col < res*0.2:
                left_flag = 1
            elif col >= res*0.4 and col <= res*0.6: 
                mid_flag = 1
            elif col > res*0.8 and col < res*0.9:
                right_flag = 1
           
    if left_flag:
        print("obstacle detected on the left")
    if right_flag:
        print("obstacle detected on the right")
    if mid_flag:
        print("obstacle detected in the middle")
    return [left_flag,mid_flag,right_flag]
        

def isParallelogram(approx):
    approx = np.array(approx)
    x_coordinates = np.sort([point[0] for point in approx])
    sorted_idx = np.argsort(approx[:,0])
    
    p0 = approx[sorted_idx[0]]
    p1 = approx[sorted_idx[1]]
    p2 = approx[sorted_idx[2]]
    p3 = approx[sorted_idx[3]]
    
    leftslope = abs((p2[1] - p0[1])/(p2[0] - p0[0]))
    rightslope = abs((p3[1] - p1[1])/(p3[0] - p1[0]))
    topslope = abs((p1[0] - p0[0])/(p1[1] - p0[1]))
    bottomslope = abs((p3[0] - p2[0])/(p3[1] - p2[1]))

    sidediff = abs(leftslope - rightslope)
    topdiff = abs(topslope - bottomslope)

    if sidediff < 0.2 and topdiff < 0.2:
        return True
    else:
        return False
    
'''ADDITIONAL FUNCTIONS'''
        
def get_area(mask, found_object, depth_path, cur_depth_name):
    cnt_img = np.zeros_like(mask)
    cnt_img = cv.fillPoly(cnt_img, [found_object], 255)
    pixelpoints = np.transpose(np.nonzero(cnt_img))
    depth = cv.imread(os.path.join(depth_path,cur_depth_name), cv.IMREAD_ANYDEPTH)
    depth = depth*1e-3

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

    return real_area
        

def compare_area(cur_obj_area, prev_obj_area):
    diff = abs(cur_obj_area - prev_obj_area)/cur_obj_area
    if diff < 0.3*cur_obj_area:
        return True
    else:
        return False
   
    
