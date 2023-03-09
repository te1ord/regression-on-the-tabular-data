import torch
import torch.nn as nn
from src.data.make_dataset import AnonymizedDataset
from torch.utils.data import DataLoader
import numpy as np



class PolynomialRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolynomialRegression, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, dataloader, num_epochs, learning_rate):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for x, y in dataloader:
                # Forward pass
                x = x.reshape(x.shape[0], 1)
                y = y.reshape(y.shape[0], 1)
                y_pred = self(x)

                # Compute loss
                loss = torch.sqrt(criterion(y_pred, y))
                epoch_loss += loss.item()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss /= len(dataloader)
            print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}")

    def predict(self, dataloader):
        with torch.no_grad():
            predictions = []
            truth_values = []
            for x, y in dataloader:

                x = x.reshape(x.shape[0], 1)
                y = y.reshape(y.shape[0], 1)

                y_pred = self(x)
                predictions.append(y_pred.detach().numpy())
                truth_values.append(y.numpy())
        return np.concatenate(predictions), np.concatenate(truth_values)




