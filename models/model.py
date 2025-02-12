# models/model.py
import torch.nn as nn
import torch.nn.functional as F

# 1. DNN: For the computed 561-feature vector input.
class HAR_DNN(nn.Module):
    def __init__(self, input_size=561, num_classes=6):
        super(HAR_DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. CNN: For raw inertial signals input with shape (batch, 9, 128)
class HAR_CNN(nn.Module):
    def __init__(self, num_classes=6):
        super(HAR_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # After two poolings: sequence length 128 -> 64 -> 32
        self.fc1 = nn.Linear(64 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x: (batch_size, 9, 128)
        x = self.pool(F.relu(self.conv1(x)))  # → (batch, 32, 64)
        x = self.pool(F.relu(self.conv2(x)))  # → (batch, 64, 32)
        x = x.view(x.size(0), -1)             # Flatten → (batch, 64*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. RNN: LSTM-based model for raw inertial signals.
#    The raw input (batch, 9, 128) is permuted to (batch, 128, 9)
class HAR_RNN(nn.Module):
    def __init__(self, num_classes=6, hidden_size=64, num_layers=2):
        super(HAR_RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=9, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch_size, 9, 128) → permute to (batch, 128, 9)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, 128, hidden_size)
        # Use the last time-step output
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
