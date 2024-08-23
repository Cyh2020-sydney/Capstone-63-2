import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    return train_dataset, test_dataset


def client_update(client_model, optimizer, train_loader, epoch=5):
    client_model.train()
    for _ in range(epoch):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = client_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
    return client_model.state_dict()


def federated_average(global_model, client_models_state_dict):
    global_state_dict = global_model.state_dict()
    for key in global_state_dict.keys():
        global_state_dict[key] = torch.stack(
            [client_models_state_dict[i][key] for i in range(len(client_models_state_dict))], 0).mean(0)
    global_model.load_state_dict(global_state_dict)
    return global_model


def federated_learning_simulation(num_clients=3, num_rounds=5):

    train_dataset, test_dataset = load_data()

    client_datasets = random_split(train_dataset, [len(train_dataset) // num_clients] * num_clients)

    global_model = SimpleCNN()

    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")
        client_models_state_dict = []

        for i in range(num_clients):
            client_model = SimpleCNN()
            client_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(client_model.parameters(), lr=0.1)
            train_loader = DataLoader(client_datasets[i], batch_size=10, shuffle=True)
            client_state_dict = client_update(client_model, optimizer, train_loader)
            client_models_state_dict.append(client_state_dict)

        global_model = federated_average(global_model, client_models_state_dict)

    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    fedtest(global_model, test_loader)


def fedtest(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset) * 100
    print(f'Test Accuracy: {accuracy:.4f}%')


federated_learning_simulation()
