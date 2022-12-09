import torch
from torch.nn.functional import pairwise_distance
from torch.utils.data.dataloader import DataLoader
import numpy as np
from sklearn.cluster import KMeans

from utils import imshow

from dataset import CIFAR10Ds
from datasetShort import CIFAR10DsShort
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()

CIFAR10Data = np.load('CIFARData.npz')['arr_0']
CIFAR10DataShort = np.load('CIFARDataShort.npz')['arr_0']
CIFAR10DataVal = np.load('CIFARValData.npz')['arr_0']

data = []
labels = []
for classno, classimg in enumerate(CIFAR10DataVal):
    for i in range(len(classimg)):
        data.append(classimg[i]/255)
        labels.append(classno)

net = torch.load('./models/siameseArchEmb10Run3.pth')
net.eval()

repData = []
rawImgs = []

count = 0
for img, label in zip(data, labels):
    count += 1

    rawImgs.append(img.flatten())

    img = torch.from_numpy(img.transpose(2, 0, 1))
    img = img.type(torch.FloatTensor)

    out = torch.squeeze(net(img)).detach().numpy()
    repData.append(out)

X = np.array(repData)
labels = np.array(labels)
kmeans = KMeans(n_clusters=4, random_state=0)
values = kmeans.fit_predict(X)

print(values[1500:1950])

# print(np.argmax(np.bincount(values[:500])), np.sum(np.where(values[:500] == np.zeros_like(values[:500])+3, 1, 0))/500)
print(np.argmax(np.bincount(values[:500])), np.sum(np.where(values[:500] == np.zeros_like(values[:500])+np.argmax(np.bincount(values[:500])), 1, 0))/500)
print(np.argmax(np.bincount(values[500:1000])), np.sum(np.where(values[500:1000] == np.zeros_like(values[500:1000])+np.argmax(np.bincount(values[500:1000])), 1, 0))/500)
print(np.argmax(np.bincount(values[1000:1500])), np.sum(np.where(values[1000:1500] == np.zeros_like(values[1000:1500])+np.argmax(np.bincount(values[1000:1500])), 1, 0))/500)
print(np.argmax(np.bincount(values[1500:])), np.sum(np.where(values[1500:] == np.zeros_like(values[1500:])+np.argmax(np.bincount(values[1500:])), 1, 0))/500)

print(values)

# X = np.array(np.array(rawImgs))
# print(X.shape)
# print(X[0].shape)
# kmeans = KMeans(n_clusters=4, random_state=0)
# values = kmeans.fit_predict(X)




