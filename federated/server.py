import flwr as fl
from flwr.server import ServerApp
from flwr.server.strategy import FedAvg

# Define strategy
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)

# Define ServerApp
app = ServerApp(
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)

if __name__ == "__main__":
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:9091",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )