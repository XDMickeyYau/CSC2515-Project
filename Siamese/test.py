import torch
from torch.nn.functional import pairwise_distance
from torch.utils.data.dataloader import DataLoader
import numpy as np

from utils import imshow

from dataset import CIFAR10Ds
from datasetShort import CIFAR10DsShort


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()

CIFAR10Data = np.load('CIFARData.npz')['arr_0']
CIFAR10DataShort = np.load('CIFARDataShort.npz')['arr_0']
CIFAR10DataTest = np.load('CIFARDataTest.npz')['arr_0']

# CifarDataset = CIFAR10DsShort(CIFAR10Data, CIFAR10DataShort, CIFAR10DataTest, train=False)
CifarDataset = CIFAR10Ds(CIFAR10Data, CIFAR10DataTest, train=False)
dataloader = DataLoader(CifarDataset, batch_size=1, shuffle=False)

net = torch.load('./models/siamese.pth')
net.eval()

for data in dataloader:

    anchors, positives, negatives = data[0], data[1], data[2]

    outAnchor = net(anchors)
    outPos = net(positives)
    outNeg = net(negatives)

    print(pairwise_distance(outAnchor, outPos))
    print(pairwise_distance(outAnchor, outNeg))

    imshow(anchors, positives, negatives)

    break



