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
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((Config['img_size'], Config['img_size']), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            'test': transforms.Compose([
                transforms.Resize((Config['img_size'], Config['img_size']), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        }
        return data_transforms


    def create_dataset(self):
        # create X, y pairs
        X_train, X_test, y_train, y_test = [],[],[],[]
        # Train data
        print('Load train data')
        dir_list = os.listdir(self.train_dir)
        for dir in tqdm(dir_list):
            for image in os.listdir(osp.join(self.train_dir, dir)):
                X_train.append(osp.join(dir, image))
                y_train.append(osp.join(dir, image[3:]))

        # Test data
        print('Load test data')
        dir_list = os.listdir(self.test_dir)
        for dir in tqdm(dir_list):
            for image in os.listdir(osp.join(self.test_dir, dir)):
                X_test.append(osp.join(dir, image))
                y_test.append(osp.join(dir, image[3:]))

        X_train, y_train = shuffle(X_train, y_train, random_state=7)
        X_test, y_test = shuffle(X_test, y_test, random_state=7)

        return X_train, X_test, y_train, y_test



class mask_train(Dataset):
    def __init__(self, X_train, y_train, transform):
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform
        self.train_dir = osp.join(Config['root_path'], 'train')
        self.y_train_dir =osp.join(Config['root_path'], 'y_train')

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, item):
        file_path1 = osp.join(self.train_dir, self.X_train[item])
        file_path2 = osp.join(self.y_train_dir, self.y_train[item])
        return self.transform(Image.open(file_path1)), self.transform(Image.open(file_path2)),


class mask_test(Dataset):
    def __init__(self, X_test, y_test, transform):
        self.X_test = X_test
        self.y_test = y_test
        self.transform = transform
        self.test_dir = osp.join(Config['root_path'], 'test')
        self.y_test_dir =osp.join(Config['root_path'], 'y_test')


    def __len__(self):
        return len(self.X_test)


    def __getitem__(self, item):
        file_path1 = osp.join(self.test_dir, self.X_test[item])
        file_path2 = osp.join(self.y_test_dir, self.y_test[item])
        return self.transform(Image.open(file_path1)), self.transform(Image.open(file_path2)),




def get_dataloader(debug, batch_size, num_workers):
    dataset = mask_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test= dataset.create_dataset()

    if debug==True:
        train_set = mask_train(X_train[:100], y_train[:100], transform=transforms['train'])
        test_set = mask_test(X_test[:100], y_test[:100], transform=transforms['test'])

        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set = mask_train(X_train, y_train, transforms['train'])
        test_set = mask_test(X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    datasets = {'train': train_set, 'test': test_set}

    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                                 for x in ['train', 'test']}
    return dataloaders, dataset_size


if __name__ == '__main__':
    dataloaders, dataset_size = get_dataloader(debug=Config['debug'],
                                               batch_size=Config['batch_size'],
                                               num_workers=Config['num_workers'])

