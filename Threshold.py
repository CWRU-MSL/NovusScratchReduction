# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:27:43 2020

@author: Nishan Senanayake
"""


import  cv2
from matplotlib import pyplot as plt
from PIL import Image 
import numpy as np
import matplotlib.patches as patches
from matplotlib import image as im
from skimage.filters import threshold_otsu, rank,threshold_minimum,threshold_local
from skimage.morphology import disk
from skimage.filters import threshold_niblack,threshold_li
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage import morphology
from sklearn.preprocessing import scale
# =============================================================================
# #Functions
# =============================================================================


# Normlizing the image
def normalize(x):
    normalized = (x-np.min(x))/(np.max(x)-np.min(x))
    return normalized 


# Crop the center of the image
def crop_center(img,new_width,new_height):
    y,x = img.shape
    startx = x//2-(new_width//2)
    starty = y//2-(new_height//2)    
    return img[starty:starty+new_height,startx:startx+new_width]

#cropping the rectangle

def crop_side(img,startx,starty,new_width,new_height):          
    return img[starty:starty+new_height,startx:startx+new_width]



def area_frac_cal(im,big_img,new_width,new_height):
    block_size = 101
    thresh = threshold_li(im)
    thresh= thresh +(thresh/6)
    binary = im > thresh
    print ('thresh = ' +str(thresh))
    # thresh =0.8
    
    # binary = im > thresh
    binary=morphology.remove_small_objects(binary, min_size=100,connectivity=0)
    scr_area_pixel= np.sum(binary==1)
    scr_Area_fraction= (float(scr_area_pixel) / float((new_width * new_height))) * 100
    return scr_Area_fraction,binary,thresh





# =============================================================================
# #Aanalysis
# =============================================================================

# #Before treatment


# #Determine the locations of  rectangles

width_bt= 1000
height_bt=1000

#vertical rectangle
x_bt=1500
y_bt=1200


dirpath='dir path'
filename = 'file_name'
filepath=dirpath+filename+'.extension'
img_before_trt_cv= cv2.imread(filepath,0)

# cv2.imwrite('Gray.png',img_before_trt_cv)
# img_before_trt_1 = im.imread(filepath)

figure, ax = plt.subplots(1)
rect = patches.Rectangle((x_bt,y_bt),width_bt,height_bt, edgecolor='r', facecolor="none")
ax.imshow(img_before_trt_cv)
ax.add_patch(rect)
plt.savefig(filename+'file_save_name')
img_before_trt_crop = crop_side(img_before_trt_cv,x_bt,y_bt,width_bt,height_bt)

normlized_img_before_trt_crop = normalize(img_before_trt_crop)
scr_area_fraction_before_trt,binary_img_before_trt ,thresh= area_frac_cal(normlized_img_before_trt_crop,img_before_trt_cv,width_bt,height_bt)


print('BT = ' + str(scr_area_fraction_before_trt) )
plt.imshow(img_before_trt_cv)
plt.imshow(img_before_trt_crop,cmap=plt.cm.gray)
plt.imshow(binary_img_before_trt,cmap=plt.cm.gray)













