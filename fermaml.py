import  torch.utils.data as data
import  os
import  os.path
import  errno
import h5py
import numpy as np
from    PIL import Image

class Fer(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'fer.training.pt'
    test_file = 'fer.test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.exists('./data/fer_maml_data.h5'):
            raise RuntimeError('Dataset not found.')
        
        self.data = h5py.File('./data/fer_maml_data.h5', 'r', driver='core')
        self.train_data = self.data['data_pixel']
        self.train_labels = self.data['data_label']
        self.train_data = np.asarray(self.train_data)
        self.train_data = self.train_data.reshape((-1, 48, 48))

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_data)



