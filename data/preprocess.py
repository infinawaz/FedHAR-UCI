# data/preprocess.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Dataset for computed features (561-dimensional vector)
class UCIHARDataset(Dataset):
    def __init__(self, data_dir, subset="train"):
        if subset == "train":
            data_path = os.path.join(data_dir, "train", "X_train.txt")
            label_path = os.path.join(data_dir, "train", "y_train.txt")
        else:
            data_path = os.path.join(data_dir, "test", "X_test.txt")
            label_path = os.path.join(data_dir, "test", "y_test.txt")
        
        self.X = np.loadtxt(data_path)
        self.y = np.loadtxt(label_path) - 1  # Convert from 1-indexed to 0-indexed
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data_loader(data_dir="data/UCI HAR Dataset", batch_size=32):
    train_dataset = UCIHARDataset(data_dir, subset="train")
    test_dataset = UCIHARDataset(data_dir, subset="test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Dataset for raw inertial signals (stacking 9 signals into shape: [n_samples, 9, 128])
class UCIHARRawDataset(Dataset):
    def __init__(self, data_dir, subset="train"):
        inertial_dir = os.path.join(data_dir, subset, "Inertial Signals")
        signal_names = [
            "body_acc_x_", "body_acc_y_", "body_acc_z_",
            "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
            "total_acc_x_", "total_acc_y_", "total_acc_z_"
        ]
        file_suffix = "train.txt" if subset == "train" else "test.txt"
        signals = []
        for signal in signal_names:
            file_path = os.path.join(inertial_dir, signal + file_suffix)
            data = np.loadtxt(file_path)  # shape: (n_samples, 128)
            signals.append(data)
        # Stack signals to get shape: (n_samples, 9, 128)
        self.X = np.stack(signals, axis=1)
        
        label_file = os.path.join(data_dir, subset, "y_" + ("train.txt" if subset == "train" else "test.txt"))
        self.y = np.loadtxt(label_file) - 1
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_raw_data_loader(data_dir="data/UCI HAR Dataset", batch_size=32):
    train_dataset = UCIHARRawDataset(data_dir, subset="train")
    test_dataset = UCIHARRawDataset(data_dir, subset="test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Quick tests (run this file directly to verify data loading)
if __name__ == "__main__":
    # Test computed features loader
    train_loader, test_loader = get_data_loader()
    print(f"Computed Features - Train dataset size: {len(train_loader.dataset)} samples")
    # Test raw signals loader
    raw_train_loader, raw_test_loader = get_raw_data_loader()
    print(f"Raw Signals - Train dataset size: {len(raw_train_loader.dataset)} samples")
