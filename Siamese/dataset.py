import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset


class CIFAR10Ds(Dataset):
    def __init__(self, data, test_data=None, train=True):
        self.train = train
        # self.data = data.reshape(-1, 32, 32, 3)
        self.data = data
        self.num_classes = 4
        
        if (test_data is not None) and (not self.train):
            self.test = test_data.reshape(-1, 32, 32, 3)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if self.train:
            anchor_class = np.random.randint(0, self.num_classes)
            anchor_class = int(idx//4500)
            negative_class = np.random.randint(0, self.num_classes) 

            while negative_class == anchor_class:
                negative_class = np.random.randint(0, self.num_classes) 
            
            # anchor = np.random.randint(0, 4500)
            positive = np.random.randint(0, 4500)
            negative = np.random.randint(0, 4500)


            anchor_img = ToTensor()(self.data[idx].reshape(32, 32, 3))
            positive_img = ToTensor()(self.data[anchor_class*4500 + positive].reshape(32, 32, 3))
            negative_img = ToTensor()(self.data[negative_class*4500 + negative].reshape(32, 32, 3)) 

        # else:
            # anchor_class = np.random.randint(0, self.num_classes)
            # negative_class = np.random.randint(0, self.num_classes) 

            # while (negative_class == anchor_class):
            #     negative_class = np.random.randint(0, self.num_classes) 

            # anchor = np.random.randint(0, 1000)
            # positive = np.random.randint(0, 1000)
            # negative = np.random.randint(0, 1000)

            # anchor_img = ToTensor()(self.test[anchor_class*1000 + anchor].reshape(32, 32, 3))
            # positive_img = ToTensor()(self.test[anchor_class*1000 + positive].reshape(32, 32, 3))
            # negative_img = ToTensor()(self.test[negative_class*1000 + negative].reshape(32, 32, 3)) 

        return [anchor_img, positive_img, negative_img]