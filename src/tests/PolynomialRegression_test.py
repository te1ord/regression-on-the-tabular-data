import torch
from torch.utils.data import TensorDataset, DataLoader
from src.model.script import PolynomialRegression
import pytest

@pytest.fixture
def dataloader():
    X = torch.arange(10).unsqueeze(1).float()
    y = 2*X + 1
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=2)

def test_forward_pass(dataloader):
    model = PolynomialRegression(input_dim=1, hidden_dim=10, output_dim=1)
    inputs, _ = next(iter(dataloader))
    outputs = model(inputs)
    assert outputs.shape == torch.Size([2, 1])

def test_train_model(dataloader):
    model = PolynomialRegression(input_dim=1, hidden_dim=10, output_dim=1)
    model.train_model(dataloader, num_epochs=10, learning_rate=0.01)
    assert model.fc1.weight.grad is not None

def test_predict(dataloader):
    model = PolynomialRegression(input_dim=1, hidden_dim=10, output_dim=1)
    y_pred, y_true = model.predict(dataloader)
    assert y_pred.shape == torch.Size([10, 1])
