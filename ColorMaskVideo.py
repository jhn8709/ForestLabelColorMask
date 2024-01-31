# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 01:31:13 2023

@author: Jimmy
"""

import cv2
import numpy as np
import pandas as pd

capture = cv2.VideoCapture("C:/Users/Jimmy/Documents/University Courses/Research Files/mp4_files/IMG_1258-720p.mp4")
total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
print(total_frames)
df = pd.read_csv('GT_IMG_1258.csv')
df2 = pd.read_csv('GT_IMG_1258_919.csv')

def drawLine(startX, startY, endX, endY):
    YPoints = []
    XPoints = []
    dx = endX - startX
    dy = endY - startY
    steps = abs(dx) if (abs(dx) > abs(dy)) else abs(dy)
    Xinc = dx / steps
    Yinc = dy / steps
    X = startX
    Y = startY
    for i in range(int(steps)):
        XPoints.append(round(X))
        YPoints.append(round(Y))
        X += Xinc
        Y += Yinc
    return XPoints, YPoints

def fillROI(GT_df, frame_index):
    
    max_width = 375
    max_y = 720 # represents floor in front
    min_y = 230 # represents "horizon"
    slope = max_width / (max_y-min_y)
    point_count = GT_df["#ofPoints"][frame_index]
    frame_points = GT_df.iloc[frame_index, 2:(2+point_count*2)]
    Xlist = []
    Ylist = []
    Xpoints = []
    Ypoints = []
    
    for i in range(0, point_count*2-2, 2):
        
        Xpoints, Ypoints = drawLine(int(frame_points[i]), int(frame_points[i+1]), 
                                     int(frame_points[i+2]), int(frame_points[i+3]))
        
        # This section determines which pixels are part of the ROI (region of interest) from the labeled points
        for j in range(len(Xpoints)):
            # This linear equation serves as a base
            width = max_width - (max_y-Ypoints[j]) * slope
            width = int(width//2)
            Xlist.append(Xpoints[j])
            Ylist.append(Ypoints[j])
            for k in range(1, width, 1):
                Xlist.append(Xpoints[j]+k)
                Ylist.append(Ypoints[j])
                Xlist.append(Xpoints[j]-k)
                Ylist.append(Ypoints[j])
                             
    return Xlist, Ylist

def applyMask(frame, Xlist, Ylist, color='green'):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    # Define the coordinates of the region of interest (roi_corners)
    roi_corners = np.array([(x, y) for x, y in zip(Xlist, Ylist)], dtype=np.int32)
    cv2.fillPoly(mask, [roi_corners], (255, 255, 255))

    # Darken the shade of the mask color (light green)
    if color == 'green':
        mask_color = (0, 175, 0)
    elif color == 'blue':
        mask_color = (175, 0, 0)
    elif color == 'red':
        mask_color = (0, 0, 175)
    #dark_green = (0, 175, 0)
    darkened_mask = cv2.bitwise_and(mask, mask, mask=mask)

    # Create an overlay with the same size as the image
    overlay = np.zeros_like(frame)
    overlay[darkened_mask != 0] = mask_color

    masked_overlay = cv2.bitwise_and(overlay, overlay, mask=mask)

    output = cv2.addWeighted(frame, 1, masked_overlay, 0.25, 0.5)
    return output

frame_index = -1
fast_forward = 1
counter = 0
capture = cv2.VideoCapture("C:/Users/Jimmy/Documents/University Courses/Research Files/mp4_files/IMG_1258-720p.mp4")
paused = False
while True:

    capture.set(cv2.CAP_PROP_FRAME_COUNT, frame_index)
    isTrue, frame = capture.read()
    if (frame_index < total_frames-2) & ((counter%fast_forward)==0):
        counter = 0
        Xlist, Ylist = fillROI(df, frame_index+1)
        Xlist2, Ylist2 = fillROI(df2, frame_index+1)
        masked_frame = applyMask(frame, Xlist, Ylist, 'blue')
        #masked_frame = applyMask(masked_frame, Xlist2, Ylist2, 'green')
        #masked_frame = applyMask(frame, Xlist2, Ylist2, 'green')
        cv2.imshow('Video', masked_frame)
        frame_index += fast_forward
        if cv2.waitKey(2) & 0xFF==ord('d'):
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(2) & 0xFF==ord('f'):
            if fast_forward == 1:
                fast_forward = 5
            else:
                fast_forward = 1
        if cv2.waitKey(2) & 0xFF==ord('g'):
            paused = not paused
            
    counter+=1