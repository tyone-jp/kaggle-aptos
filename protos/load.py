import pandas as pd
import numpy as np

TRAIN_PATH='../input/train.csv'
TEST_PATH='../input/test.csv'

def load_test():
    df=pd.read_csv(TEST_PATH)
    return df

def load_train():
    df=pd.read_csv(TRAIN_PATH)
    return df

if __name__=='__main__':
    print(load_test().head())
    print(load_train().head())
    
