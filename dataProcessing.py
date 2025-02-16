from matplotlib import pyplot as plt
import numpy as np
import h5py
import os

activityIDdict = {
    1: 'walking',
    2: 'walking_upstairs',
    3: 'walking_downstairs',
    4: 'sitting',
    5: 'standing',
    0: 'laying',
}
colNames = ['body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z']


def read_files():
    list_of_Xs = ['./test/Inertial Signals',
                  './train/Inertial Signals']

    list_of_ys = ['./test/y_test.txt',
                  './train/y_train.txt']

    train_X_array = []
    files = os.listdir(list_of_Xs[1])
    for file in files:
        print(file, " is reading...")
        data = np.loadtxt(list_of_Xs[1] + '/' + file)
        train_X_array.append(data)

    test_X_array = []
    files = os.listdir(list_of_Xs[0])
    for file in files:
        print(file, " is reading...")
        data = np.loadtxt(list_of_Xs[0] + '/' + file)
        test_X_array.append(data)

    train_X = np.dstack(train_X_array)
    test_X = np.dstack(test_X_array)

    train_y = np.loadtxt(list_of_ys[1])
    test_y = np.loadtxt(list_of_ys[0])

    X = np.vstack((train_X, test_X))
    y = np.hstack((train_y, test_y)).astype(int)
    y = np.where(y == 6, 0, y)

    return [X, y]


def partition_data(x, y, num_clients=5):
    data_per_client = len(y) // num_clients
    client_datasets = []

    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client if i != num_clients - 1 else len(y)
        client_X = x[start_idx:end_idx]
        client_y = y[start_idx:end_idx]
        client_datasets.append((client_X, client_y))

    return client_datasets


def save_data(arr, file_name):
    dict_ = {'inputs': arr[0], 'labels': arr[1]}
    f = h5py.File(file_name, 'w')
    for key in dict_:
        f.create_dataset(key, data=dict_[key])
    f.close()
    print('Done.')


def save_federated_data(client_datasets):
    for i, (client_X, client_y) in enumerate(client_datasets):
        file_name = f'client_{i + 1}_data.h5'
        with h5py.File(file_name, 'w') as f:
            f.create_dataset('inputs', data=client_X)
            f.create_dataset('labels', data=client_y)
        print(f'Data saved for Client {i + 1} in {file_name}')


def window_plot(X, y, col, y_index):
    unit = 'ms^-2'
    x_seq = X[y_index][:, col]
    plot_title = colNames[col] + ' - ' + activityIDdict[y[y_index]]
    plt.plot(x_seq)
    plt.title(plot_title)
    plt.xlabel('window')
    plt.ylabel(unit)
    plt.show()


if __name__ == "__main__":
    arr = read_files()
    num_clients = 5
    client_datasets = partition_data(arr[0], arr[1], num_clients)
    save_federated_data(client_datasets)