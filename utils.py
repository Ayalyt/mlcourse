import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class DataProcessor():
    def __init__(self):
        
        self.device = torch.device('cuda')
        self.fashion_train_df = pd.read_csv('./data/fashion-mnist_train.csv', sep=',')
        self.fashion_test_df = pd.read_csv('./data/fashion-mnist_test.csv', sep=',')
        self.training = np.asarray(self.fashion_train_df, dtype='float32')
        self.testing = np.asarray(self.fashion_test_df, dtype='float32')
    
        self.X_train = self.training[:, 1:].reshape([-1,28,28,1]) / 255
        self.y_train = self.training[:, 0]
        
        self.X_test = self.testing[:, 1:].reshape([-1,28,28,1]) / 255
        self.y_test = self.testing[:, 0]
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=12345)
        self.X_train = self.X_train.reshape(48000, 1, 28, 28)
        self.X_val = self.X_val.reshape(12000,1,28,28)
        self.X_test = self.X_test.reshape(10000, 1, 28, 28)
        
    def get_data_loaders(self, batch_size=64):



        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.long).to(self.device)

        X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(self.y_val, dtype=torch.long).to(self.device)

        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.long).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
        
