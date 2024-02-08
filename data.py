import pandas as pd
import matplotlib.pyplot as plt

#global train, test
train: pd.DataFrame = None
test: pd.DataFrame = None
 
def load():
    global train
    train = pd.read_csv('data/1_taxi_train.csv')
    print(f'train shape {train.shape}')
    print(train.head())

    # Read the test data
    global test
    test = pd.read_csv('data/1_taxi_test.csv')

    # Print train and test columns
    print('Train columns:', train.columns.tolist())
    print('Test columns:', test.columns.tolist())