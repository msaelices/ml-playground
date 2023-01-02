import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


from ..mnist import MNISTModel, SDGOptimizer


def test_mnist_training():
    train_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )
    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MNISTModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SDGOptimizer(model.parameters(), lr=1e-3)

    loss, X, y, pred = next(model.train_gen(train_dataloader, loss_fn, optimizer))

    assert model.training is True
    assert loss.device.type == device
    assert X.device.type == device
    assert y.device.type == device
    assert pred.device.type == device
    assert loss.item() > 0
    assert X.shape == (batch_size, 1, 28, 28)
    assert y.shape == (batch_size,)
    assert pred.shape == (batch_size, 10)
    assert pred.argmax(1).shape == (batch_size,)
    assert pred.argmax(1).dtype == torch.int64
    assert pred.argmax(1).min() >= 0
    assert pred.argmax(1).max() <= 9


def test_mnist_testing():
    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    batch_size = 64
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MNISTModel().to(device)
    loss_fn = nn.CrossEntropyLoss()

    loss, correct = model.test_gen(test_dataloader, loss_fn)

    assert model.training is False
    assert isinstance(loss, float)
    assert isinstance(correct, float)
    assert loss > 0.0
    assert correct >= 0.0
    assert correct <= 1.0
