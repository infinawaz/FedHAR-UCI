import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_data_loader(data_dir, batch_size=32):
    # Load precomputed features (DNN version)
    X_train = np.loadtxt(f"{data_dir}/train/X_train.txt")
    y_train = np.loadtxt(f"{data_dir}/train/y_train.txt", dtype=int) - 1
    X_test = np.loadtxt(f"{data_dir}/test/X_test.txt")
    y_test = np.loadtxt(f"{data_dir}/test/y_test.txt", dtype=int) - 1

    # Convert to PyTorch tensors
    train_data = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))
    
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(test_data, batch_size=batch_size)
    )

def get_raw_data_loader(data_dir, batch_size=32):
    # Load raw inertial signals (CNN/RNN version)
    def load_signals(path):
        signals = []
        for axis in ["x", "y", "z"]:
            for sensor in ["acc", "gyro"]:
                file = f"{path}/Inertial Signals/{sensor}_body_{axis}_train.txt"
                signals.append(np.loadtxt(file))
        return np.stack(signals, axis=1)  # [samples, 9, 128]
    
    # Load and process data
    X_train = load_signals(f"{data_dir}/train")
    y_train = np.loadtxt(f"{data_dir}/train/y_train.txt", dtype=int) - 1
    X_test = load_signals(f"{data_dir}/test")
    y_test = np.loadtxt(f"{data_dir}/test/y_test.txt", dtype=int) - 1

    train_data = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))
    
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(test_data, batch_size=batch_size)
    )