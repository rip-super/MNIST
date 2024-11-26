import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def load_data():
    train_data = datasets.MNIST(
        root="data",
        train=True,
        transform=ToTensor(),
        download=True
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        transform=ToTensor(),
        download=True
    )

    loaders = {
        "train": DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
        "test": DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),
    }
    
    return loaders

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def train(model, device, loaders, optimizer, loss_func, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 60 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders['train'].dataset)} "
                  f"({100. * batch_idx / len(loaders['train']):.0f}%)]\t{loss.item():.6f}")

def test(model, device, loaders, loss_func):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders["test"].dataset)
    print(f"\nTest Set: Average Loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['test'].dataset)} "
          f"({100. * correct / len(loaders['test'].dataset):.0f}%)\n")

def test_prediction(model, test_data, idx, total_tests, device):
    data, target = test_data[idx]
    data = data.unsqueeze(0).to(device)
    output = model(data)
    prediction = output.argmax(dim=1, keepdim=True).item()
    image = data.squeeze(0).squeeze(0).cpu().numpy()

    plt.figure()
    plt.imshow(image, cmap="gray", interpolation='nearest')
    plt.title(f"Prediction: {prediction}")
    plt.show(block=(idx == total_tests - 1))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = load_data()
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        train(model, device, loaders, optimizer, loss_func, epoch)
        test(model, device, loaders, loss_func)

    total_tests = 4
    for i in range(total_tests):
        test_prediction(model, loaders["test"].dataset, i, total_tests, device)

if __name__ == "__main__":
    main()