import pandas as pd
import numpy as np
from load import load_train,load_test,load_train1
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

df_train=load_train1()

y_train=pd.get_dummies(df_train[:3510*1]['level']).values

x=np.load('../input/diabetic-retinopathy-resized/circle_cropping/circle_cropping0.npz')
x=x['x_train']
x_train=x.astype(np.uint8)

x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.15,random_state=0)

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
        model.add(layers.Dense(5,activation='softmax'))

        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.00005),metrics=['accuracy'])
        return model

class Metrics(Callback):
        def on_train_begin(self,logs={}):
            self.val_kappas=[]

        def on_epoch_end(self,epoch,logs={}):
            X_val,y_val=self.validation_data[:2]
            y_pred=self.model.predict(X_val)
            _val_kappa=cohen_kappa_score(y_val.argmax(axis=1),
                                          y_pred.argmax(axis=1),
                                          weights='quadratic')

            self.val_kappas.append(_val_kappa)

            print(f'val_kappa:{_val_kappa:.4f}')
            return
    
model=build_model()
model.summary()

kappa_metrics=Metrics()
checkpoint=ModelCheckpoint('pre_trained_model1.h5',monitor='val_loss',
                                                      verbose=0,save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto')
history=model.fit_generator(data_generator,
                             steps_per_epoch=x_train.shape[0]/BATCH_SIZE,
                             epochs=15,
                             validation_data=(x_val,y_val),
                             callbacks=[checkpoint,kappa_metrics])
 
 
