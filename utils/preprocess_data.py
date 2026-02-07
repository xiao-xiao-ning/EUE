import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import os

np.random.seed(42)  # For reproducibility
random.seed(42)
torch.set_num_threads(32)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = 'data'

def preprocess_data(data_name:str, normalize:bool, train_ratio=0.8, val_ratio=0.1):
    print("Processing ...", data_name, "£"*29, normalize)
    file_path_train = f'./datasets/{data_name}/{data_name}_TRAIN.txt'  
    file_path_test = f'./datasets/{data_name}/{data_name}_TEST.txt'
    df_1 = pd.read_csv(file_path_train, header=None, delim_whitespace=True).dropna()
    df_2 = pd.read_csv(file_path_test, header=None, delim_whitespace=True).dropna()

    # Separate labels and data
    labels_1 = df_1.iloc[:, 0].values
    data_1 = df_1.iloc[:, 1:].values
    labels_2 = df_2.iloc[:, 0].values
    data_2 = df_2.iloc[:, 1:].values
    
    print('Before Processing Training Labels', np.unique(labels_1))
    print('Before Processing Validation Labels', np.unique(labels_2))
    # exit()


    if data_name in ['OliveOil', 'TwoPatterns', 'CinCECGTorso', 'EthanolLevel']: # 1, 2, 3, 4 -> 0, 1, 2, 3
        # print(labels_1)
        # print(labels_1 - 1)
        # exit()
        labels_1 = labels_1 - 1
        labels_2 = labels_2 - 1 
    elif data_name in ['CBF']: # 1, 2, 3 -> 0, 1, 2
        labels_1 = labels_1 - 1
        labels_2 = labels_2 - 1  
    elif data_name in ['FordA', 'FordB', 'ECG200', 'Wafer']: # 1, -1 -> 1, 0
        labels_1[labels_1==-1] = 0
        labels_2[labels_2==-1] = 0 
    elif data_name in ['GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Strawberry', 'Yoga', 'Chinatown', 'DodgerLoopGame', 'TwoLeadECG', 'FreezerRegularTrain', 'SemgHandGenderCh2', 'FreezerSmallTrain']: # 1, 2 -> 0, 1
        labels_1 = labels_1 - 1
        labels_2 = labels_2 - 1  
    elif data_name in ['MixedShapesRegularTrain', 'BME', 'MixedShapesSmallTrain', 'SemgHandSubjectCh2']: # 1, 2, 3, 4, 5 = 0, 1, 2, 3, 4
        labels_1 = labels_1 - 1
        labels_2 = labels_2 - 1 
    elif data_name in ['TwoPatterns', 'ArrowHead', 'Earthquakes', 'HandOutlines']:
        labels_1 = labels_1
        labels_2 = labels_2
    else:
        print("No Data")
        exit(1)

    data = np.concatenate((data_1, data_2))
    labels = np.concatenate((labels_1, labels_2))[:, np.newaxis]

    XY = np.concatenate((data, labels), axis=1)
    print(data.shape, labels.shape, XY.shape)
    np.random.shuffle(XY)
    data = XY[:, :-1]  # All rows, all columns except the last
    labels = XY[:, -1]   # All rows, last column

    # Define split ratio
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)

    train_data, val_data,  test_data = data[:train_size], data[train_size:(train_size+val_size)], data[(train_size+val_size):]
    train_labels, val_labels, test_labels = labels[:train_size],  labels[train_size:(train_size+val_size)], labels[(train_size+val_size):]

    if len(train_data) + len(val_data) + len(test_data) != len(data) or  len(train_labels) + len(val_labels) + len(test_labels) != len(data):
        print("Data not match!")
        exit(1)
    
    print('After Processing Training Labels', np.unique(train_labels))
    print('After Processing Validation Labels', np.unique(val_labels))
    print('After Processing Testing Labels', np.unique(test_labels))

    if normalize:
        mean = np.mean(train_data)
        std = np.std(train_data)
        train_data = (train_data - mean)/std
        val_data = (val_data - mean)/std
        test_data = (test_data - mean)/std


    data_dir = os.path.join(root_dir, data_name)
    os.makedirs(data_dir, exist_ok=True)

    train_data_file = os.path.join(data_dir, 'train_data.npy')
    val_data_file = os.path.join(data_dir, 'val_data.npy')
    test_data_file = os.path.join(data_dir, 'test_data.npy')


    train_labels_file = os.path.join(data_dir, 'train_labels.npy')
    val_labels_file = os.path.join(data_dir, 'val_labels.npy')
    test_labels_file = os.path.join(data_dir, 'test_labels.npy')

    print(train_data.shape, train_labels.shape)

    np.save(train_data_file, train_data)
    np.save(val_data_file, val_data)
    np.save(test_data_file, test_data)
    np.save(train_labels_file, train_labels)
    np.save(val_labels_file, val_labels)
    np.save(test_labels_file, test_labels)





