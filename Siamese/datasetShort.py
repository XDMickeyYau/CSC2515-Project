import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset


class CIFAR10DsShort(Dataset):
    def __init__(self, data, short, test_data=None, train=True):
        self.train = train
        self.data = data.reshape(-1, 32, 32, 3)
        self.short = short.reshape(-1, 32, 32, 3)
        self.num_classes = 4

        if test_data is not None:
            self.test = test_data.reshape(-1, 32, 32, 3)
 
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        if self.train:
            anchor_class = np.random.randint(0, self.num_classes) 
            negative_class = np.random.randint(0, self.num_classes) 

            # while (negative_class == anchor_class):
            #     negative_class = np.random.randint(0, self.num_classes) 

            anchor = np.random.randint(0, 500)
            positive = np.random.randint(0, 500)
            negative = np.random.randint(0, 500)

            anchor_img = ToTensor()(self.short[anchor_class*500 + anchor].reshape(32, 32, 3))
            positive_img = ToTensor()(self.short[anchor_class*500 + positive].reshape(32, 32, 3))
            negative_img = ToTensor()(self.data[negative_class*500 + negative].reshape(32, 32, 3)) 
        
        else:
            anchor_class = np.random.randint(0, self.num_classes)
            negative_class = np.random.randint(0, self.num_classes) 

            while (negative_class == anchor_class):
                negative_class = np.random.randint(0, 10) 

            anchor = np.random.randint(0, 1000)
            positive = np.random.randint(0, 1000)
            negative = np.random.randint(0, 1000)

            anchor_img = ToTensor()(self.test[anchor_class*1000 + anchor].reshape(32, 32, 3))
            positive_img = ToTensor()(self.test[anchor_class*1000 + positive].reshape(32, 32, 3))
            negative_img = ToTensor()(self.test[negative_class*1000 + negative].reshape(32, 32, 3)) 

        return [anchor_img, positive_img, negative_img]