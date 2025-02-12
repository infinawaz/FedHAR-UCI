# federated/client.py
import argparse
import flwr as fl
import torch
from models.model import HAR_DNN, HAR_CNN, HAR_RNN
from models.train import train, test
from data.preprocess import get_data_loader, get_raw_data_loader

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, epochs=1, device=self.device)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.test_loader, device=self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

def main():
    parser = argparse.ArgumentParser(description="Flower client for HAR dataset")
    parser.add_argument("--model", type=str, default="dnn", choices=["dnn", "cnn", "rnn"],
                        help="Model type to use: dnn, cnn, or rnn")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    args = parser.parse_args()
    
    if args.model == "dnn":
        # Use computed features (561-dim vector)
        train_loader, test_loader = get_data_loader(data_dir="data/UCI HAR Dataset", batch_size=args.batch_size)
        model = HAR_DNN(input_size=561, num_classes=6)
    else:
        # Use raw inertial signals (shape: [batch, 9, 128])
        train_loader, test_loader = get_raw_data_loader(data_dir="data/UCI HAR Dataset", batch_size=args.batch_size)
        if args.model == "cnn":
            model = HAR_CNN(num_classes=6)
        elif args.model == "rnn":
            model = HAR_RNN(num_classes=6)
    
    client = FlowerClient(model, train_loader, test_loader, device=args.device)
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)

if __name__ == "__main__":
    main()
