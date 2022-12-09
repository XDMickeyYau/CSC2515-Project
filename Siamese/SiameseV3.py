from torch import nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 5)
        self.conv4 = nn.Conv2d(16, 32, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 4)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(-1, 32 * 3 * 3)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


cnn = CNN()

img = torch.rand(1, 3, 32, 32)
print(cnn.pool(cnn.conv4(cnn.conv3(cnn.pool(cnn.conv2(cnn.conv1(img)))))).shape)
# print(cnn.pool(cnn.conv4(cnn.conv3(cnn.pool(cnn.conv2(cnn.conv1(img)))))).shape)

