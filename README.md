# Federated Learning for HAR using Flower and PyTorch

This repository implements a Federated Learning (FL) framework using [Flower](https://flower.dev/) and [PyTorch](https://pytorch.org/) for a Human Activity Recognition (HAR) task. The code distributes training across multiple clients, each training on local data stored in HDF5 files, and aggregates the model updates on a central server using the FedAvg strategy.

## Features

- **Federated Learning:** Distributed training where each client trains on its own dataset.
- **Dynamic CNN Architecture:** The CNN dynamically computes its flattened size based on input dimensions.
- **Detailed Logging:** Clients print epoch-level training losses, and the server logs aggregated evaluation metrics.
- **Visualization:** After training, the server generates charts and tables showing global accuracy and loss over rounds.
- **Customizable Hyperparameters:** Easily adjust global rounds, local epochs, learning rate, batch size, and more.

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [Flower](https://flower.dev/)
- [h5py](https://www.h5py.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)

Install the required packages using pip:

```bash
pip install torch flwr h5py numpy matplotlib pandas
Setup & Usage
1. Prepare Client Data
For each client, create an HDF5 file (e.g., client_1_data.h5) with two datasets:

inputs: The input data (e.g., images or sensor readings) stored as a NumPy array.
labels: The corresponding labels stored as a NumPy array.
Ensure the data shapes are compatible with the CNN. For example:

Grayscale images: (num_samples, height, width)
Images with channels: (num_samples, channels, height, width)
2. Running the Server
Start the Flower server (which is configured to run for 10 rounds by default):

bash
Copy
Edit
python har_fl.py --server
3. Running the Clients
In separate terminal windows, start each client by providing its unique client ID. For example:

bash
Copy
Edit
python har_fl.py --client 1
python har_fl.py --client 2
python har_fl.py --client 3
python har_fl.py --client 4
python har_fl.py --client 5
4. Monitoring Training
Client Terminals: Each client prints the average loss per epoch during local training.
Server Terminal: The server prints global evaluation metrics (accuracy and loss) for each round. At the end of training, the server displays charts and a summary table.
Hyperparameter Tuning
Global Rounds: Adjust the number of rounds by modifying ServerConfig(num_rounds=10) in the server initialization.
Local Epochs: Change the number of local epochs in the client's fit() method.
Learning Rate & Batch Size: These can be modified in the optimizer settings and DataLoader configuration, respectively.
Troubleshooting
Warnings: You may see warnings about missing metric aggregation functions. These can be suppressed by providing dummy aggregation functions in the server strategy (see the code for details).
Data Shapes: Verify that your HDF5 files contain correctly shaped data. For grayscale images, the data should have shape (num_samples, height, width); if already including a channel dimension, it should be (num_samples, channels, height, width).
Connectivity: Ensure the server is running before starting the clients, and that the server and client addresses match.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Flower for the federated learning framework.
PyTorch for deep learning support.
pgsql
Copy
Edit
