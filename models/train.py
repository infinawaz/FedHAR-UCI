# models/train.py
import torch
import torch.nn as nn
import torch.optim as optim

def train(model, train_loader, epochs=1, lr=0.01, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

def test(model, test_loader, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.to(device)
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(target).sum().item()
            total += data.size(0)
    
    avg_loss = test_loss / total
    accuracy = correct / total
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    return avg_loss, accuracy
