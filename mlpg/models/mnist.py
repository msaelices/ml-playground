import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from typing import Generator


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # layer to flatten the 28x28 image input into a 784 vector
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),  # ReLU activation function
            nn.Linear(512, 512),
            nn.ReLU(),  # ReLU activation function
            nn.Linear(512, 10),  # 10 output classes, with the probability of each digit [0-9]
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        """Forward pass of the model"""
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

    def train_gen(self, dataloader: DataLoader, loss_fn: nn.Module, optimizer: Optimizer) -> Generator:
        """Train the model returning a generator with each batch loss, input, output and prediction"""
        self.train()
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self(X)
            loss = loss_fn(pred, y)

            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            yield loss, X, y, pred

    def test_gen(self, dataloader: DataLoader, loss_fn: nn.Module) -> tuple[float, int]:
        """Test the model returning a generator with each loss value and the correct predictions"""
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()
        test_loss, correct = 0.0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        return test_loss, correct
