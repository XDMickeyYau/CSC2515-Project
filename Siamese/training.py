import torch
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
import numpy as np

from Siamese import CNN
from utils import imshow, triplet_loss

from dataset import CIFAR10Ds
from datasetShort import CIFAR10DsShort


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()

CIFAR10Data = np.load('CIFARData.npz')['arr_0'].reshape(-1, 32, 32, 3)
CIFAR10DataShort = np.load('CIFARDataShort.npz')['arr_0'].reshape(-1, 32, 32, 3)

CifarDataset = CIFAR10Ds(CIFAR10Data)
dataloader = DataLoader(CifarDataset, batch_size=128, shuffle=False)
net = CNN()
optimizer = Adam(net.parameters(), lr=5e-4)

epochs = 35
for i in range(epochs):

    print('\n EPOCH #{} \n'.format(i+1))

    batch_loss = 0

    for batchno, batch in enumerate(dataloader):
        anchors, positives, negatives = batch[0], batch[1], batch[2]

        optimizer.zero_grad()

        outAnchor = net(anchors)
        outPos = net(positives)
        outNeg = net(negatives)

        loss = triplet_loss(1, [outAnchor, outPos, outNeg])
        loss.sum().backward()

        optimizer.step()

        batch_loss += loss.mean().item()

        if batchno % 20 == 0:
            print(batch_loss/20)
            batch_loss = 0

torch.save(net, './models/siameseArchEmb50Run3.pth')