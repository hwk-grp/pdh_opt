import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(dim_in, 64)
        self.dp1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dp2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(32, dim_out)

    def forward(self, x):
        h = self.dp1(torch.tanh(self.fc1(x)))
        h = self.dp2(F.relu(self.fc2(h)))
        out = self.fc3(h)

        return out


def train(model, data_loader, optimizer, criterion):
    model.train()
    sum_losses = 0

    for data, targets in data_loader:
        data = data.cpu()
        targets = targets.cpu()

        preds = model(data)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_losses += loss.item()

    return sum_losses / len(data_loader)


def test(model, data_loader):
    model.eval()
    list_preds = list()

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.cpu()
            list_preds.append(model(data).cpu().numpy())

    return numpy.vstack(list_preds)
