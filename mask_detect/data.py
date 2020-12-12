import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
import os
import numpy as np
import os.path as osp
from tqdm import tqdm
from PIL import Image
from utils import Config
from PIL import Image


class mask_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.train_dir = osp.join(self.root_dir, 'train')
        self.y_train_dir =osp.join(self.root_dir, 'y_train')
        self.test_dir = osp.join(self.root_dir,'test')
        self.y_test_dir = osp.join(self.root_dir, 'y_test')
        self.transforms = self.get_data_transforms()


    def get_data_transforms(self):
        data_transforms = transforms.Compose([
                transforms.Resize((256, 256), Image.BICUBIC),
                transforms.ToTensor(),
            ])
        return data_transforms


    def create_dataset(self):
        # create X, y pairs
        X = []
        y = []
        # Train data
        print('Load train data')
        dir_list = os.listdir(self.y_train_dir)
        for dir in tqdm(dir_list):
            for image in os.listdir(osp.join(self.y_train_dir, dir)):
                X.append(osp.join('train', dir, '01_'+image))
                y.append(osp.join(dir, image))

        # Test data
        print('Load test data')
        dir_list = os.listdir(self.y_test_dir)
        for dir in tqdm(dir_list):
            for image in os.listdir(osp.join(self.y_test_dir, dir)):
                X.append(osp.join('test', dir, '01_'+image))
                y.append(osp.join(dir, image))
        return X, y



class mask(Dataset):
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform
        self.root = Config['root_path']


    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        file_path = osp.join(self.root, self.X[item])
        return self.transform(Image.open(file_path).convert("RGB")), self.y[item]



def get_dataloader(debug, batch_size, num_workers):
    dataset = mask_dataset()
    transforms = dataset.get_data_transforms()
    X, y= dataset.create_dataset()

    if debug==True:
        data_set = mask(X[:10], y[:10], transform=transforms)
    else:
        data_set = mask(X, y, transforms)


    dataloaders = DataLoader(data_set,
                                 shuffle=False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
    return dataloaders


if __name__ == '__main__':
    dataloader = get_dataloader(debug=Config['debug'],
                                               batch_size=Config['batch_size'],
                                               num_workers=Config['num_workers'])

