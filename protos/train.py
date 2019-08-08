import pandas as pd
import numpy as np
from load import load_train,load_test
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score,accuracy_score
from tqdm import tqdm
from mixgenerator import MixupGenerator

from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam

df_train=load_train()
df_test=load_test()

y_train=pd.get_dummies(df_train['diagnosis']).values
y_train_multi=np.empty(y_train.shape,dtype=y_train.dtype)
y_train_multi[:,4]=y_train[:,4]
for i in range(3,-1,-1):
    y_train_multi[:,i]=np.logical_or(y_train[:,i],y_train_multi[:,i+1])

data_file='../input/aptos2019-blindness-detection/circle_cropping.npz'
data=np.load(data_file)
x_train=data['x_train'].astype(int)
x_test=data['x_test'].astype(int)

x_train,x_val,y_train,y_val=train_test_split(x_train,y_train_multi,test_size=0.15,random_state=0)

BATCH_SIZE=32

def create_datagen():
    return ImageDataGenerator(zoom_range=0.15,fill_mode='constant',
                              cval=0.,
                              horizontal_flip=True,
                              vertical_flip=True)
data_generator=create_datagen().flow(x_train,y_train,batch_size=BATCH_SIZE)
mixup_generator=MixupGenerator(x_train,y_train,batch_size=BATCH_SIZE,alpha=0.2,datagen=create_datagen())()

densenet=DenseNet121(weights='../input/weight/DenseNet-BC-121-32-no-top.h5',
                                         include_top=False,input_shape=(224,224,3))
def build_model():
        model=Sequential()
        model.add(densenet)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(5,activation='sigmoid'))

        model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.00005),metrics=['accuracy'])
        return model

class Metrics(Callback):
        def on_train_begin(self,logs={}):
            self.val_kappas=[]

        def on_epoch_end(self,epoch,logs={}):
            X_val,y_val=self.validation_data[:2]
            y_val=y_val.sum(axis=1)-1
            y_pred=self.model.predict(X_val)>0.5
            y_pred=y_pred.astype(int).sum(axis=1)-1
            _val_kappa=cohen_kappa_score(y_val,
                                          y_pred,
                                          weights='quadratic')

            self.val_kappas.append(_val_kappa)

            print(f'val_kappa:{_val_kappa:.4f}')

            if _val_kappa==max(self.val_kappas):
                print('Vlidation Kappa has improved. Saveing model.')
                self.model.save('model.h5')
            return
        
    
model=build_model()
model.summary()

kappa_metrics=Metrics()
history=model.fit_generator(data_generator,
                             steps_per_epoch=x_train.shape[0]/BATCH_SIZE,
                             epochs=15,
                             validation_data=(x_val,y_val),
                             callbacks=[kappa_metrics])
 
model.load_weights('model.h5')
y_pred=model.predict(x_train)>0.5
y_pred=y_pred.astype(int).sum(axis=1)-1
cohen_kappa=cohen_kappa_score(y_pred,df_train['diagnosis'].values,weight='quadratic')
print(cohen_kappa)
