import argparse
from typing import Dict, List, Tuple

import flwr as fl
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class HARCNN(nn.Module):
    def __init__(self, in_channels: int = 1, input_size: Tuple[int, int] = (28, 28)):
        """
        CNN model that computes the flattened feature size dynamically.
        :param in_channels: Number of channels in the input images.
        :param input_size: Tuple (H, W) of the input image size.
        """
        super(HARCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.2)

        # Use a dummy input to compute the flattened feature size.
        dummy_input = torch.zeros(1, in_channels, *input_size)
        x = torch.relu(self.conv1(dummy_input))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        self.flatten_size = x.view(1, -1).shape[1]
        print(f"Computed flatten_size from dummy input: {self.flatten_size}")

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 6)  # Adjust the number of output classes if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If x doesn't have a channel dimension, add one.
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class HARClient(fl.client.NumPyClient):
    def __init__(self, cid: int):
        self.cid = cid

        # Load client-specific data from HDF5
        with h5py.File(f'client_{cid}_data.h5', 'r') as f:
            X = np.array(f['inputs'])
            y = np.array(f['labels'])
        print(f"Loaded data shape for client {cid}: {X.shape}")

        # Determine input channels and image size from data
        if len(X.shape) == 3:
            # Data shape: (num_samples, H, W)
            in_channels = 1
            input_size = X.shape[1:]  # (H, W)
        elif len(X.shape) == 4:
            # Data shape: (num_samples, C, H, W)
            in_channels = X.shape[1]
            input_size = X.shape[2:]
        else:
            raise ValueError(f"Unsupported data shape: {X.shape}")

        # Initialize model using the actual data dimensions
        self.model = HARCNN(in_channels=in_channels, input_size=input_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Convert data to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        self.trainloader = DataLoader(
            TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True
        )

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(5):  # Local epochs
            for X, y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        loss_total, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss_total += criterion(outputs, y).item()
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = 100.0 * correct / total
        return loss_total, len(self.trainloader.dataset), {"accuracy": accuracy}


def start_server():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,      # Use all available clients for training
        fraction_evaluate=0.5,
        min_fit_clients=5,
        min_evaluate_clients=2,
        min_available_clients=5,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


def start_client(cid: int):
    client = HARClient(cid)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true", help="Start Flower server")
    parser.add_argument("--client", type=int, help="Client ID")
    args = parser.parse_args()

    if args.server:
        start_server()
    elif args.client is not None:
        start_client(args.client)
