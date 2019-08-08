import os
import pandas as pd
import numpy as np
from load import load_train1
from modify import gaussianblur
from tqdm import tqdm
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam

from logging import StreamHandler,DEBUG,Formatter,FileHandler,getLogger

logger=getLogger(__name__)
DIR='result/tmp'

def gaussianblur(img,sigmaX=10):
    x=np.empty((img.shape))
    for i in range(img.shape[0]):
        x[i]=cv2.addWeighted(img[i],4,cv2.GaussianBlur(img[i],(0,0),sigmaX),-4,128)
    return x

def create_datagen():
    return ImageDataGenerator(zoom_range=0.15,
                              fill_mode='constant',
                              cval=0.,
                              horizontal_flip=True,
                              vertical_flip=True)
densenet=DenseNet121(weights='../input/weight/DenseNet-BC-121-32-no-top.h5',
                     include_top=False,input_shape=(224,224,3))

def build_model():
    model=Sequential()
    model.add(densenet)
    model.add(layers.GlobalAvgPool2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5,activation='sigmoid'))
           
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.00005),metrics=['accuracy'])
    return model

class Metrics(Callback):
    def on_train_begin(self,logs={}):
        self.val_kappas=[]
                                                            
    def on_epoch_end(self,epoch,logs={}):
        x_val,y_val=self.validation_data[:2]
        y_val=y_val.sum(axis=1)-1
        y_pred=self.model.predict(x_val)>0.5
        y_pred=y_pred.astype(int).sum(axis=1)-1
        _val_cohen_kappa_score=cohen_kappa_score(y_val,y_pred,weights='quadratic')
        self.val_kappas.append(_val_cohen_kappa_score)
                    
        print(f'val_kappa:{_val_cohen_kappa_score:.4f}')

        if _val_cohen_kappa_score==max(self.val_kappas):
            print('Validation kappa improved. Saving model')
            self.model.save('pre_trained.h5')
        return
                            
if __name__=='__main__':
    log_fmt=Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s] [%(funcName)s] %(message)s')
    handler=StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler=FileHandler(DIR+'train.py.log','a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df_train=load_train1()

    file='../input/diabetic-retinopathy-resized/circle_cropping/circle_cropping'

    for i in range(3):
        y_train=pd.get_dummies(df_train[3*i*3510:3*(i+1)*3510]['level']).values
        y_train_multi=np.empty((y_train.shape))
        y_train_multi[:,4]=y_train[:,4]
        for j in range(3,-1,-1):
            y_train_multi[:,j]=np.logical_or(y_train[:,j],y_train_multi[:,j+1])
        logger.info('y shape:{}'.format(y_train_multi.shape))

        x_train=np.empty((3510*3,224,224,3))
        for k in range(3):
            x=np.load(file+str(i*3+k)+'.npz')
            x=x['x_train']
            x=gaussianblur(x)
            x_train[k*3510:(k+1)*3510,:,:,:]=x
        x_train=x_train.astype(np.uint8)
        logger.info('x shape:{}'.format(x_train.shape))

        x_train,x_val,y_train,y_val=train_test_split(x_train,y_train_multi,test_size=0.15,random_state=0)
        logger.info('x_train,val shape:{} {}'.format(x_train.shape,x_val.shape))
        logger.info('y_train,val shape:{} {}'.format(y_train.shape,y_val.shape))
        
        BATCH_SIZE=32

        data_generator=create_datagen().flow(x_train,y_train,batch_size=BATCH_SIZE)

        model=build_model()
        if os.path.exists('pre_trained.h5'):
            model.load_weights('pre_trained.h5')
            
        kappa_metrics=Metrics()

        histry=model.fit_generator(data_generator,
                                   steps_per_epoch=x_train.shape[0]/BATCH_SIZE,
                                   epochs=15,
                                   validation_data=(x_val,y_val),
                                   callbacks=[kappa_metrics])

        model.load_weights('pre_trained.h5')
        y_pred=model.predict(x_train)>0.5
        y_pred=y_pred.astype(int).sum(axis=1)-1
        cohen_kappa=cohen_kappa_score(y_pred,y_train,weights='quadratic')
        logger.debug('cohen_kappa:{}'.format(cohen_kappa))
                                      
        

        
        

        
