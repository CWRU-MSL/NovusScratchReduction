# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:08:18 2020

@author: 14198
"""


 
import numpy as np
import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import glob
import os
import matplotlib.patches as patches


def normalize(x):
    normalized = (x-np.min(x))/(np.max(x)-np.min(x))
    return normalized 

def feature_extraction(img):
    df = pd.DataFrame()


#All features generated must match the way features are generated for TRAINING.
#Feature1 is our original image pixels
    img2 = img.reshape(-1)
    df['Original Image'] = img2

#Generate Gabor features
    num = 1
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
#               print(theta, sigma, , lamda, frequency)
                
                    gabor_label = 'Gabor' + str(num)
#                    print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter image and add values to new column
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Modify this to add new column for each gabor
                    num += 1
########################################
#Geerate OTHER FEATURES and add them to the data frame
#Feature 3 is canny edge
    edges = cv2.Canny(img, 200,250)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe

    from skimage.filters import roberts, sobel, scharr, prewitt

#Feature 4 is Roberts edge
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1

#Feature 5 is Sobel
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

#Feature 6 is Scharr
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    #Feature 7 is Prewitt
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1

    #Feature 8 is Gaussian with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    #Feature 9 is Gaussian with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

    #Feature 10 is Median with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1

    #Feature 11 is Variance with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1  #Add column to original dataframe
    
    orb = cv2.ORB_create(20000)
    kp, des = orb.detectAndCompute(img, None)
    orb_img = cv2.drawKeypoints(img, kp, None, flags=None)
    orb_img= cv2.cvtColor(orb_img,cv2.COLOR_BGR2GRAY)
    orb_img = orb_img.reshape(-1)
    df['ORB'] = orb_img 
    
    kmeans = KMeans(n_clusters=3, random_state=0).fit(img)
    kmeans_lbl = kmeans.cluster_centers_[kmeans.labels_]
    kmeans_lbl = kmeans_lbl.reshape(-1)
    df['kmeans'] = kmeans_lbl 
    
    # thresh = threshold_li(img)
    # binary = img > thresh
    # binary = binary.reshape(-1)
    # df['binary'] = binary

    return df



#creating the dataframe
df1=pd.DataFrame()
df2=pd.DataFrame()
mask_df= pd.DataFrame()
img_df= pd.DataFrame()
final_df= pd.DataFrame()
for directory_path in glob.glob("Images/Training/Images_set2"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path,0)
        df1= feature_extraction(img)
        img_df= img_df.append(df1)

for directory_path in glob.glob("Images/Training/Masks_set2"):
    for mask_path  in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)  
        label_img = mask.reshape(-1)
        df2['Label_Value'] = label_img
        mask_df= mask_df.append(df2)




final_df= pd.concat([img_df, mask_df], axis=1) 

print(final_df.head())


Y = final_df["Label_Value"].values  # Lables
X=  final_df.drop(labels = ["Label_Value"], axis=1)  # Features



model = RandomForestClassifier(n_estimators = 100, random_state = 30)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model.fit(X_train, Y_train)
prediction_test = model.predict(X_test)



print ("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)


#testing
from skimage import morphology
def areafrac(binary):
    
    scr_area_pixel= np.sum(binary==1)
    scr_Area_fraction= (float(scr_area_pixel) / float((binary.shape[0]* binary.shape[0]))) * 100
    return scr_Area_fraction,binary


def crop_side(img,startx,starty,new_width,new_height):          
    return img[starty:starty+new_height,startx:startx+new_width]



# #Determine the locations of  rectangles

width_bt= 1000
height_bt=1000

#vertical rectangle
x_bt=1700
y_bt=900




dirpath='S1_BT_NewCAB/'
filename = '13D_BT'
filepath=dirpath+filename+'.tif'
img_before_trt_cv= cv2.imread(filepath,0)

# cv2.imwrite('Gray.png',img_before_trt_cv)
# img_before_trt_1 = im.imread(filepath)

figure, ax = plt.subplots(1)
rect = patches.Rectangle((x_bt,y_bt),width_bt,height_bt, edgecolor='r', facecolor="none")
ax.imshow(img_before_trt_cv)
ax.add_patch(rect)
plt.savefig(filename+'_ana1_BT.png')
img_before_trt_crop = crop_side(img_before_trt_cv,x_bt,y_bt,width_bt,height_bt)
normlized_img_before_trt_crop = normalize(img_before_trt_crop)

predict= feature_extraction(img_before_trt_crop)


prediction_test = model.predict(predict)
segmented = prediction_test.reshape((img_before_trt_crop.shape))
plt.imshow(img_before_trt_crop)
plt.imshow(segmented)


scr_area_fraction ,binary= areafrac(segmented)
print(scr_area_fraction)
