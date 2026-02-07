import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from pytorch_wavelets import DWT1DForward, DWT1DInverse
import json

random.seed(42)
torch.set_num_threads(32)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the custom dataset
class TimeseriesDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (list/array): The input data.
            labels (list/array): The corresponding labels for the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetch a single data point and its label.
        """
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

def load_data(data_name:str):
    train_data = np.load(f'./data/{data_name}/train_data.npy')
    train_labels = np.load(f'./data/{data_name}/train_labels.npy')
    val_data = np.load(f'./data/{data_name}/val_data.npy')
    val_labels = np.load(f'./data/{data_name}/val_labels.npy')
    test_data = np.load(f'./data/{data_name}/test_data.npy')
    test_labels = np.load(f'./data/{data_name}/test_labels.npy')
    mean_json = f'./data/{data_name}/mean_dict.json'

    # Load the dictionary from the JSON file
    with open(mean_json, "r") as json_file:
        mean_dict = json.load(json_file)

    print(np.unique(test_labels), np.unique(train_labels))

    train_data_tensor = torch.tensor(train_data).unsqueeze(1).float()
    train_labels_tensor = torch.tensor(train_labels).long()
    val_data_tensor = torch.tensor(val_data).unsqueeze(1).float()
    val_labels_tensor = torch.tensor(val_labels).long()
    test_data_tensor = torch.tensor(test_data).unsqueeze(1).float()
    test_labels_tensor = torch.tensor(test_labels).long()

    print("Train Data Shape", train_data_tensor.shape)
    print("Train Labels Shape", train_labels_tensor.shape)
    print("Validation Data Shape", val_data_tensor.shape)
    print("Validation Labels Shape", val_labels_tensor.shape)
    print("Test Data Shape", test_data_tensor.shape)
    print("Test Labels Shape", test_labels_tensor.shape)


    train_dataset = TimeseriesDataset(train_data_tensor, train_labels_tensor)
    val_dataset = TimeseriesDataset(val_data_tensor, val_labels_tensor)
    test_dataset = TimeseriesDataset(test_data_tensor, test_labels_tensor)

    return train_dataset, val_dataset, test_dataset, mean_dict

def load_noise_data(data_name:str):
    train_data = np.load(f'./data/{data_name}/train_data.npy')
    train_labels = np.load(f'./data/{data_name}/train_labels.npy')
    val_data = np.load(f'./data/{data_name}/val_data.npy')
    val_labels = np.load(f'./data/{data_name}/val_labels.npy')
    test_data = np.load(f'./data/{data_name}/test_data.npy')
    test_labels = np.load(f'./data/{data_name}/test_labels.npy')
    mean_json = f'./data/{data_name}/mean_dict.json'

    # Load the dictionary from the JSON file
    with open(mean_json, "r") as json_file:
        mean_dict = json.load(json_file)

    print(np.unique(test_labels), np.unique(train_labels))
    noise = np.random.normal(loc=0, scale=0.1, size=test_data.shape)
    # print(noise)
    # exit()
    test_data = test_data + noise

    train_data_tensor = torch.tensor(train_data).unsqueeze(1).float()
    train_labels_tensor = torch.tensor(train_labels).long()
    val_data_tensor = torch.tensor(val_data).unsqueeze(1).float()
    val_labels_tensor = torch.tensor(val_labels).long()
    test_data_tensor = torch.tensor(test_data).unsqueeze(1).float()
    test_labels_tensor = torch.tensor(test_labels).long()

    print("Train Data Shape", train_data_tensor.shape)
    print("Train Labels Shape", train_labels_tensor.shape)
    print("Validation Data Shape", val_data_tensor.shape)
    print("Validation Labels Shape", val_labels_tensor.shape)
    print("Test Data Shape", test_data_tensor.shape)
    print("Test Labels Shape", test_labels_tensor.shape)


    train_dataset = TimeseriesDataset(train_data_tensor, train_labels_tensor)
    val_dataset = TimeseriesDataset(val_data_tensor, val_labels_tensor)
    test_dataset = TimeseriesDataset(test_data_tensor, test_labels_tensor)

    return train_dataset, val_dataset, test_dataset, mean_dict
       

def load_data_withproperties(data_name:str):
    train_data = np.load(f'./data/{data_name}/train_data.npy')
    train_labels = np.load(f'./data/{data_name}/train_labels.npy')
    # test_data = np.load(f'./data/{data_name}/test_data.npy')
    # test_labels = np.load(f'./data/{data_name}/test_labels.npy')

    mean_dict = {}
    mean = train_data.mean()
    std = train_data.std()
    mean_series = torch.ones((1, 1, train_data.shape[-1]))*mean
    mean_dict[0] = mean_series

    for wavelet_level in range(9):
        dwt1d = DWT1DForward(wave='haar', J=wavelet_level)
        yl, yh = dwt1d(mean_series)
        wave_input = [yl] + yh
        dwt_test_instance = torch.cat(wave_input, dim=-1)
        mean_dict[wavelet_level] = dwt_test_instance.flatten().numpy().tolist()

    # print(np.unique(test_labels), np.unique(train_labels))

    train_data_tensor = torch.tensor(train_data)
    train_data_tensor = train_data_tensor.unsqueeze(1).float()
    train_labels_tensor = torch.tensor(train_labels).long()
    # test_data_tensor = torch.tensor(test_data).unsqueeze(1).float()
    # test_labels_tensor = torch.tensor(test_labels).long()

    print("Train Data Shape", train_data_tensor.shape)
    print("Train Labels Shape", train_labels_tensor.shape)
    # print("Test Data Shape", test_data_tensor.shape)
    # print("Test Labels Shape", test_labels_tensor.shape)


    train_dataset = TimeseriesDataset(train_data_tensor, train_labels_tensor)
    # test_dataset = TimeseriesDataset(test_data_tensor, test_labels_tensor)

    mean_file = f'./data/{data_name}/mean_dict.json'
    # Save dictionary to a JSON file
    with open(mean_file, 'w') as json_file:
        json.dump(mean_dict, json_file, indent=4)


    return train_dataset
