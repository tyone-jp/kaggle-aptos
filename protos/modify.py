import pandas as pd
import numpy as np
from load_data import load_train_data,load_test_data
from PIL import Image
from tqdm import tqdm

df_train=load_train_data()
df_test=load_test_data()

def preprocessing(img,size=224):
    img=Image.open(img)
    img=img.resize((size,size),Image.LANCZOS)


N_train=df_train.shape[0]
N_test=df_test.shape[0]
x_train=np.empty((N_train,224,224,3))
x_test=np.empty((N_test,224,224,3))
for i,path in enumerate(tqdm(df_train['id_code'])):
    x_train[i,:,:,:]=preprocessing(f'../input/aptos2019-blindness-detection/train_images/{path}.png')
for i,path in enumerate(tqdm(df_test['id_code'])):
    x_test[i,:,:,:]=preprocessing(f'../input/aptos2019-blindness-detection/test_images/{path}.png')

np.savez('../input/aptos2019-blindness-detection/resize_test_train',x_train=x_train,x_test=x_test)

