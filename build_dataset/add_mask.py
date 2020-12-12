2# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 20:59:48 2018

@author: msi
"""

import numpy as np
import cv2
import dlib
import csv
import os
from tqdm import tqdm


path_pic = "report/"
save_pic = "rep/"
label_pic = "re/"
# print(os.listdir(path_pic))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# cv2读取图像


for file in tqdm(os.listdir(path_pic)):
    os.makedirs(save_pic+file)
    os.makedirs(label_pic+file)
    for image in os.listdir(path_pic+file+'/'):
        
        img = cv2.imread(path_pic+file+'/'+image)    
        # 取灰度
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 人脸数rects
        rects = detector(img_gray, 0)
        if(len(rects)!=0):
            cv2.imwrite(label_pic+file+'/'+image, img)
            imgblack = img.copy()
            imgwhite = img.copy()
            for i in range(len(rects)):
                landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
                dst_pts = np.array(
                    [
                        landmarks[1],
                        landmarks[2],
                        landmarks[3],
                        landmarks[4],
                        landmarks[5],
                        landmarks[6],
                        landmarks[7],
                        landmarks[8],
                        landmarks[9],
                        landmarks[10],
                        landmarks[11],
                        landmarks[12],
                        landmarks[13],
                        landmarks[14],
                        landmarks[15],
                        landmarks[29],
                    ],
                    dtype="float32",
                )
                mask_annotation = "annotation.csv"
                with open(mask_annotation) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=",")
                    src_pts = []
                    for i, row in enumerate(csv_reader):
                        # skip head or empty line if it's there
                        try:
                            src_pts.append(np.array([float(row[1]), float(row[2])]))
                        except ValueError:
                            continue
                src_pts = np.array(src_pts, dtype="float32")
                if (landmarks > 0).all():
                    # load mask image
                    mask_img = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)
                    mask_img = mask_img.astype(np.float32)
                    mask_img = mask_img / 255.0
                    # get the perspective transformation matrix
                    M, _ = cv2.findHomography(src_pts, dst_pts)
                    
                    # transformed masked image
                    transformed_mask = cv2.warpPerspective(
                        mask_img,
                        M,
                        (img.shape[1], img.shape[0]),
                        None,
                        cv2.INTER_LINEAR,
                        cv2.BORDER_CONSTANT,
                    )
                    # mask overlay
                    alpha_mask = transformed_mask[:, :,3]
                    alpha_image = 1.0 - alpha_mask
                    
                    for c in range(0, 3):
                        img[:, :, c] = (
                            alpha_mask * transformed_mask[:, :, c]*200 +
                            alpha_image * img[:, :, c]
                        ) 
                        imgblack[:, :, c] = (
                            alpha_mask * transformed_mask[:, :, c]*20 +
                            alpha_image * img[:, :, c]
                        ) 
                        imgwhite[:, :, c] = (
                            alpha_mask * transformed_mask[:, :, 0]*200 +
                            alpha_image * img[:, :, c]
                        ) 
        
            cv2.imwrite(save_pic+file+'/'+'01_'+image, img)
            cv2.imwrite(save_pic+file+'/'+'02_'+image, imgblack)
            cv2.imwrite(save_pic+file+'/'+'03_'+image, imgwhite)