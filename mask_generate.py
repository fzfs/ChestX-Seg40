import os
import cv2
import numpy as np
import pandas as pd

img_size = 512
fixed_size = [img_size,img_size,3]

# change path ['train', 'test']
path_npy = 'train/mask/'
path_csv = 'train/train.csv'
df = pd.read_csv(path_csv)

for index, row in df.iterrows():    
    scapula = eval(row['scapula'])
    vertebra = eval(row['vertebra'])
    rib = eval(row['rib'])
    lung = eval(row['lung'])
    heart = eval(row['heart'])
    trachea = eval(row['trachea'])
    clavicle = eval(row['clavicle'])
    
    # scapula
    mask_scapula = np.zeros((len(scapula), img_size, img_size))
    for j in range(len(scapula)):            
        temp = np.zeros(fixed_size)
        pts_scapula = np.array(scapula[j])/4
        temp_scapula = cv2.fillPoly(temp, np.int32([pts_scapula]), (1, 1, 1))
        mask_scapula[j] = temp_scapula[:, :, 0]
    
    # thoracic vertebrae
    mask_vertebra = np.zeros((len(vertebra), img_size, img_size))
    for j in range(len(vertebra)):            
        temp = np.zeros(fixed_size)
        pts_vertebra = np.array(vertebra[j])/4
        temp_vertebra = cv2.fillPoly(temp, np.int32([pts_vertebra]), (1, 1, 1))
        mask_vertebra[j] = temp_vertebra[:, :, 0]
    
    # ribs
    mask_rib = np.zeros((len(rib), img_size, img_size))
    for j in range(len(rib)):            
        temp = np.zeros(fixed_size)
        pts_rib = np.array(rib[j])/4
        temp_rib = cv2.fillPoly(temp, np.int32([pts_rib]), (1, 1, 1))
        mask_rib[j] = temp_rib[:, :, 0]
            
    # lungs
    mask_lung = np.zeros((len(lung), img_size, img_size))
    for j in range(len(lung)):            
        temp = np.zeros(fixed_size)
        pts_lung = np.array(lung[j])/4
        temp_lung = cv2.fillPoly(temp, np.int32([pts_lung]), (1, 1, 1))
        mask_lung[j] = temp_lung[:, :, 0]
    
    # heart
    mask_heart = np.zeros((1, img_size, img_size))          
    temp = np.zeros(fixed_size)
    pts_heart = np.array(heart)/4
    temp_heart = cv2.fillPoly(temp, np.int32([pts_heart]), (1, 1, 1))
    mask_heart[0] = temp_heart[:, :, 0]
    
    # trachea
    mask_trachea = np.zeros((1, img_size, img_size))          
    temp = np.zeros(fixed_size)
    pts_trachea = np.array(trachea)/4
    temp_trachea = cv2.fillPoly(temp, np.int32([pts_trachea]), (1, 1, 1))
    mask_trachea[0] = temp_trachea[:, :, 0]
    
    # clavicles
    mask_clavicle = np.zeros((len(clavicle), img_size, img_size))
    for j in range(len(clavicle)):            
        temp = np.zeros(fixed_size)
        pts_clavicle = np.array(clavicle[j])/4
        temp_clavicle = cv2.fillPoly(temp, np.int32([pts_clavicle]), (1, 1, 1))
        mask_clavicle[j] = temp_clavicle[:, :, 0]
    
    target = np.zeros((40, img_size, img_size), np.float16)
    target[0:2] = mask_scapula
    target[2:14] = mask_vertebra
    target[14:34] = mask_rib
    target[34:36] = mask_lung
    target[36:37] = mask_heart
    target[37:38] = mask_trachea
    target[38:40] = mask_clavicle
    
    # save
    np.save(os.path.join(path_npy,row['filename']),target)