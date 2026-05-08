import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# Function for training a single epoch
def train(model, dataloader, criterion, optimizer, learning_rule, device):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()


# Function for testing the model
def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.squeeze(), target)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    return total_loss / total_samples


def train_and_test(model, train_loader, test_loader, optimizer, learning_rule, problem_type='classification', epochs=20, device = None):
    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if problem_type == 'classification':
        if model.out_dim == 1:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
    elif problem_type == 'regression':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Invalid problem_type. Choose either 'classification' or 'regression'.")

    training_errors = []
    test_errors = []
    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, learning_rule, device)
        train_error = test(model, train_loader, criterion, device)
        test_error = test(model, test_loader, criterion, device)
        training_errors.append(train_error)
        test_errors.append(test_error)
    return training_errors, test_errors

