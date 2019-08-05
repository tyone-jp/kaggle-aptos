import numpy as np
from sklearn.metrics import cohen_kappa_score
from load import load_train
import pandas as pd

from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam

df_train=load_train()
y_train=df_train['diagnosis'].values

x=np.load('../input/aptos2019-blindness-detection/circle_cropping.npz')
x_train=x['x_train']

densenet=DenseNet121(weights='../input/weight/DenseNet-BC-121-32-no-top.h5',include_top=False,input_shape=(224,224,3))

def build_model():
    model=Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5,activation='sigmoid'))
    return model

model=build_model()
model.load_weights('model.h5')

y_pred=model.predict(x_train)>0.5
y_pred=y_pred.astype(int).sum(axis=1)-1

cohen_kappa=cohen_kappa_score(y_train,y_pred,weights='quadratic')
print(cohen_kappa)
