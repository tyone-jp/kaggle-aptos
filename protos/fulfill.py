from tqdm import tqdm
from load import load_train,load_test
import pandas as pd
import numpy as np
from modify import circle_crop

df_train=load_train()
df_test=load_test()

save='../input/aptos2019-blindness-detection/circle_cropping.npz'
modify_train=circle_crop

N_train=df_train.shape[0]
N_test=df_test.shape[0]
x_train=np.empty((N_train,224,224,3))
x_test=np.empty((N_test,224,224,3))
for i,path in enumerate(tqdm(df_train['id_code'])):
    x_train[i,:,:,:]=modify_train(f'../input/aptos2019-blindness-detection/train_images/{path}.png')
for i,path in enumerate(tqdm(df_test['id_code'])):
    x_test[i,:,:,:]=modify_train(f'../input/aptos2019-blindness-detection/test_images/{path}.png')

np.savez(save,x_train=x_train,x_test=x_test)
