import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

input = torch.randn(1, 1, 32, 32)
out = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(out, target)

net.zero_grad()
print(net.conv1.bias.grad)

loss.backward()
print(net.conv1.bias.grad)

params = list(net.parameters())
print(params[0][0])
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
print(params[0][0])

r = torch.rand((7, 7))
p = torch.rand((1, 7))
print(torch.argmax(r@p.T))

torch.manual_seed(42)
r = torch.rand((1, 7))
p = torch.rand((1, 7))
print(r, p)
# tensor([[0.8823, 0.9150, 0.3829, 0.9593, 0.3904, 0.6009, 0.2566]])
# tensor([[0.7936, 0.9408, 0.1332, 0.9346, 0.5936, 0.8694, 0.5677]])
