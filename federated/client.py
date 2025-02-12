import argparse
import flwr as fl
import torch
from models.model import HAR_DNN, HAR_CNN, HAR_RNN
from models.train import train, test
from data.preprocess import get_data_loader, get_raw_data_loader

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, 
              epochs=config.get("local_epochs", 1), 
              device=self.device)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.test_loader, self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

def client_fn(cid: str) -> FlowerClient:
    """Client creation function for Flower's Virtual Client Engine"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dnn", choices=["dnn", "cnn", "rnn"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.model == "dnn":
        train_loader, test_loader = get_data_loader(
            data_dir="data/UCI HAR Dataset", 
            batch_size=args.batch_size
        )
        model = HAR_DNN(input_size=561, num_classes=6)
    else:
        train_loader, test_loader = get_raw_data_loader(
            data_dir="data/UCI HAR Dataset", 
            batch_size=args.batch_size
        )
        model = HAR_CNN(num_classes=6) if args.model == "cnn" else HAR_RNN(num_classes=6)

    return FlowerClient(model, train_loader, test_loader, args.device)

# Flower 1.0+ ClientApp
app = fl.client.ClientApp(
    client_fn=client_fn,
)

if __name__ == "__main__":
    # Parse arguments manually
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dnn", choices=["dnn", "cnn", "rnn"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Instantiate client
    if args.model == "dnn":
        train_loader, test_loader = get_data_loader(data_dir="data/UCI HAR Dataset", batch_size=args.batch_size)
        model = HAR_DNN(input_size=561, num_classes=6)
    else:
        train_loader, test_loader = get_raw_data_loader(data_dir="data/UCI HAR Dataset", batch_size=args.batch_size)
        model = HAR_CNN(num_classes=6) if args.model == "cnn" else HAR_RNN(num_classes=6)

    client = FlowerClient(model, train_loader, test_loader, args.device)
    
    # Start client
    fl.client.start_client(
        server_address="127.0.0.1:9091",
        client=client.to_client()
    )