import torch.nn as nn

class HAR_DNN(nn.Module):
    def __init__(self, input_size=561, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

class HAR_CNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(128*30, num_classes)  # Adjust based on input length
    
    def forward(self, x):
        x = self.cnn(x)
        return self.classifier(x)

class HAR_RNN(nn.Module):
    def __init__(self, num_classes=6, hidden_size=128):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=9,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        return self.classifier(hidden[-1])