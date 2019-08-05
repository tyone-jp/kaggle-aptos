from tqdm import tqdm
import pandas as pd
import numpy as np
from load import load_train1
from modify import circle_crop

save_file='../input/diabetic-retinopathy-resized/circle_cropping'
df_train=load_train1()
N_train=int(df_train.shape[0]/10)

x_train=np.empty((N_train,224,224,3))
i=0

while True:
    for j,img in enumerate(tqdm(df_train['image'][i*N_train:(i+1)*N_train])):
        x_train[j,:,:]=circle_crop(f'../input/diabetic-retinopathy-resized/resized_train_cropped/{img}.jpeg')
    np.savez(save_file+str(i)+'.npz',x_train=x_train)
    i+=1
    if i>9:
        break


