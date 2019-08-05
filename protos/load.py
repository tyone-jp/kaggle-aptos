import pandas as pd
import numpy as np

TRAIN_PATH='../input/aptos2019-blindness-detection/train.csv'
TEST_PATH='../input/aptos2019-blindness-detection/test.csv'
TRAIN_PATH1='../input/diabetic-retinopathy-resized/trainLabels_cropped.csv'

def load_test():
    df=pd.read_csv(TEST_PATH)
    return df

def load_train():
    df=pd.read_csv(TRAIN_PATH)
    return df

def load_train1():
    df=pd.read_csv(TRAIN_PATH1)
    return df

