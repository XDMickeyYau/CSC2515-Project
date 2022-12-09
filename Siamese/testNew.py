import torch
from torch.nn.functional import pairwise_distance
from torch.utils.data.dataloader import DataLoader
import numpy as np

from utils import imshow

from dataset import CIFAR10Ds
from datasetShort import CIFAR10DsShort
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()

CIFAR10Data = np.load('CIFARData.npz')['arr_0']
CIFAR10DataShort = np.load('CIFARDataShort.npz')['arr_0']
CIFAR10DataTest = np.load('CIFARDataTest.npz')['arr_0']

print(CIFAR10DataTest.shape)
# CIFAR10DataTest = CIFAR10DataTest.reshape(-1, 32, 32, 3)


data = []
labels = []
for classno, classimg in enumerate(CIFAR10DataTest):
    for i in range(100):
        data.append(classimg[i]/255)
        labels.append(classno)

# data = np.array(data)
# labels = np.array(labels)

# CifarDataset = CIFAR10DsShort(CIFAR10Data, CIFAR10DataShort, CIFAR10DataTest, train=False)
# CifarDataset = CIFAR10Ds(CIFAR10Data, CIFAR10DataTest, train=False)
# dataloader = DataLoader(CifarDataset, batch_size=1, shuffle=False)

net = torch.load('./models/siameseArchEmb4Run1.pth')
net.eval()
# for data in dataloader:

repData = []
rawImgs = []

count = 0
for img, label in zip(data, labels):
    count += 1
    # anchors, positives, negatives = data[0], data[1], data[2]

    rawImgs.append(img.flatten())

    img = torch.from_numpy(img.transpose(2, 0, 1))
    img = img.type(torch.FloatTensor)

    out = torch.squeeze(net(img)).detach().numpy()
    repData.append(out)

    # if count == 39:
    #     img0 = img
    #     img0e = out

    # if count == 49:
    #     img1 = img
    #     img1e = out

    # if count == 159:
    #     img2 = img
    #     img2e = out

    #     break



from sklearn.cluster import KMeans



X = np.array(repData)
labels = np.array(labels)
print(X.shape)
print(X[0].shape)
kmeans = KMeans(n_clusters=4, random_state=0)
values = kmeans.fit_predict(X)


print(np.argmax(np.bincount(values[:100])), np.sum(np.where(values[:100] == np.zeros_like(values[:100])+np.argmax(np.bincount(values[:100])), 1, 0)))
print(np.argmax(np.bincount(values[100:200])), np.sum(np.where(values[100:200] == np.zeros_like(values[100:200])+np.argmax(np.bincount(values[100:200])), 1, 0)))
print(np.argmax(np.bincount(values[200:300])), np.sum(np.where(values[200:300] == np.zeros_like(values[200:300])+np.argmax(np.bincount(values[200:300])), 1, 0)))
print(np.argmax(np.bincount(values[300:])), np.sum(np.where(values[300:] == np.zeros_like(values[300:])+np.argmax(np.bincount(values[300:])), 1, 0)))



print(values)

plt.scatter(X[:, 0], X[:, 1], c=values)
plt.show()

X = np.array(np.array(rawImgs))
print(X.shape)
print(X[0].shape)
kmeans = KMeans(n_clusters=4, random_state=0)
values = kmeans.fit_predict(X)

print(values)

# print(kmeans.labels_)

# print(np.where(kmeans.labels_ == labels, 1, 0))

# print(pairwise_distance(img0e, img1e))
# print(pairwise_distance(img0e, img2e))
# imshow(img0, img1, img2)

# ax = fig.add_subplot(1, 3, 1)
# ax.imshow(img0)
# ax.set_title("Anchor")

# ax = fig.add_subplot(1, 3, 2)
# ax.imshow(img1)
# ax.set_title("Positive")

# ax = fig.add_subplot(1, 3, 3)
# ax.imshow(img2)
# ax.set_title("Negative")

