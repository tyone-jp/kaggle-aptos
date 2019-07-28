import pandas as pd
import numpy as np
from load import load_train,load_test
import cv2
from tqdm import tqdm

df_train=load_train()
df_test=load_test()
IMAGE_SIZE=224

def preprocessing(img,sigmaX=10):
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

N_train=df_train.shape[0]
N_test=df_test.shape[0]
x_train=np.empty((N_train,224,224,3))
x_test=np.empty((N_test,224,224,3))
for i,path in enumerate(tqdm(df_train['id_code'])):
    x_train[i,:,:,:]=preprocessing(f'../input/aptos2019-blindness-detection/train_images/{path}.png')
for i,path in enumerate(tqdm(df_test['id_code'])):
    x_test[i,:,:,:]=preprocessing(f'../input/aptos2019-blindness-detection/test_images/{path}.png')

np.savez('../input/aptos2019-blindness-detection/cropping_blur.npz',x_train=x_train,x_test=x_test)

