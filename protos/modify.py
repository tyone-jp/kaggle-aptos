import pandas as pd
import numpy as np
from load import load_train,load_test
import cv2
from tqdm import tqdm

df_train=load_train()
df_test=load_test()
IMAGE_SIZE=224

def gaussianblur(img,sigmaX=10):
    img=cv2.imread(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=crop_image_from_gray(img)
    img=cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
    img=cv2.addWeighted(img,4,cv2.GaussianBlur(img,(0,0),sigmaX),-4,128)
    return img

def crop_image_from_gray(img,tol=7):
    gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    mask=gray_img>7
    check_shape=img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
    if check_shape==0:
        return img
    else:
        img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
        img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
        img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
        img=np.stack([img1,img2,img3],axis=-1)
    return img

def circle_crop(img):
    img=cv2.imread(img)
    img=crop_image_from_gray(img)
    height,width,depth=img.shape

    largest_size=np.max((height,width))
    img=cv2.resize(img,(largest_size,largest_size))
    height,width,depth=img.shape

    x=int(width/2)
    y=int(height/2)
    r=np.amin((x,y))

    circle_image=np.zeros((height,width),np.uint8)
    cv2.circle(circle_image,(x,y),int(r),1,thickness=-1)
    img=cv2.bitwise_and(img,img,mask=circle_image)
    img=crop_image_from_gray(img)
    img=cv2.resize(img,(224,224))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

