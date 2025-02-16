# Federated Learning for HAR using Flower and PyTorch

This repository implements a Federated Learning (FL) framework using [Flower](https://flower.dev/) and [PyTorch](https://pytorch.org/) for a Human Activity Recognition (HAR) task. The code distributes training across multiple clients, each training on local data stored in HDF5 files, and aggregates the model updates on a central server using the FedAvg strategy.

## Features

- **Federated Learning:** Distributed training where each client trains on its own dataset.
- **Dynamic CNN Architecture:** The CNN dynamically computes its flattened feature size based on the input dimensions.
- **Detailed Logging:** Clients print epoch-level training losses and the server logs aggregated evaluation metrics.
- **Visualization:** After training, the server generates charts and a summary table showing global accuracy and loss over rounds.
- **Customizable Hyperparameters:** Easily adjust global rounds, local epochs, learning rate, batch size, and more.

## Requirements

- Python 3.8+
- Install required packages with:

  ```bash
  pip install -r requirements.txt

- [PyTorch](https://pytorch.org/)
- [Flower](https://flower.dev/)
- [h5py](https://www.h5py.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)

Install the required packages with:

```bash
pip install torch flwr h5py numpy matplotlib pandas

Repository Structure

Below is an example requirements.txt file that you can include in your repository. Simply create a file named requirements.txt with the following content:

## requirements.txt
torch
flwr
h5py
numpy
matplotlib
pandas

.
.
├── har_fl.py                # Main federated learning code (server and client)
├── data_processing.py       # Script for processing raw HAR data and generating client datasets
├── requirements.txt         # File listing required Python packages
├── client_1_data.h5         # Generated HDF5 file for client 1 (contains "inputs" and "labels")
├── client_2_data.h5         # Generated HDF5 file for client 2
├── client_3_data.h5         # Generated HDF5 file for client 3
├── client_4_data.h5         # Generated HDF5 file for client 4
├── client_5_data.h5         # Generated HDF5 file for client 5
└── README.md                # This file



Setup & Usage

1. Data Processing

The repository includes a data_processing.py script that reads, partitions, and saves raw HAR data into HDF5 files for each client. It contains the following functions:

read_files()

Reads raw data files from the train and test directories (under the Inertial Signals subfolder) and combines them into a single dataset. Activity labels are loaded from y_train.txt and y_test.txt, with label 6 remapped to 0.


partition_data(x, y, num_clients=5)

Splits the dataset into equal parts for a specified number of clients (default is 5).


save_data(arr, file_name)
Saves a given dataset (tuple of inputs and labels) to an HDF5 file.


save_federated_data(client_datasets)
Saves each client's partitioned data into separate HDF5 files (e.g., client_1_data.h5, client_2_data.h5, etc.).


window_plot(X, y, col, y_index)
Plots a window of sensor data for a specific sensor column and activity. Useful for data visualization and debugging.


How to Use the Data Processing Script
Prepare your raw data:
Organize your data as follows:

bash


.
├── train
│   ├── Inertial Signals
│   │   ├── file1.txt
│   │   ├── file2.txt
│   │   └── ... 
│   └── y_train.txt
└── test
    ├── Inertial Signals
    │   ├── file1.txt
    │   ├── file2.txt
    │   └── ...
    └── y_test.txt


Run the script:
Execute the script to generate federated HDF5 files for each client:

bash

python data_processing.py


The script will:

Read the raw data from the train and test directories.
Partition the data into 5 clients.
Save each partition as client_1_data.h5, client_2_data.h5, ..., client_5_data.h5.


Visualize Data (Optional):
To visualize a specific sensor channel for a given activity window, use the window_plot function in an interactive session:

python

from data_processing import read_files, window_plot, colNames, activityIDdict
X, y = read_files()
# Plot the data for column index 0 (e.g., 'body_acc_x') for the first sample
window_plot(X, y, col=0, y_index=0)


2. Running the Server

Start the Flower server. By default, it is configured to run for 10 rounds:

bash
python har_fl.py --server

3. Running the Clients
In separate terminal windows, start each client by providing its unique client ID. For example:

bash

python har_fl.py --client 1
python har_fl.py --client 2
python har_fl.py --client 3
python har_fl.py --client 4
python har_fl.py --client 5

4. Monitoring Training

    Client Terminals: Each client prints the average loss per epoch during local training.
    Server Terminal: The server prints global evaluation metrics (accuracy and loss) for each round. After training, charts and a summary table will be displayed.

Hyperparameter Tuning
    Global Rounds: Modify the number of rounds by changing ServerConfig(num_rounds=10) in the server initialization.

    Local Epochs: Adjust the number of local epochs in the client's fit() method.

    Learning Rate & Batch Size: Change these parameters in the optimizer and DataLoader configurations.


License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Flower for the federated learning framework.
PyTorch for deep learning support.
