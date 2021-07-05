import numpy as np
from .dataset_base import BaseDataSet
from utils.data_io import *
import os
import tarfile
import subprocess
import os
import urllib.request

class CIFARDataSet(BaseDataSet):
    def __init__(self, dataset_path,
                 img_width=32, img_height=32, num_images=None, train=True,
                 shuffle=False, low=-1, high=1):
        super.__init__(self, dataset_path, img_height, img_width)
        self.images, self.attributes = maybe_download_cifar(dataset_path, train)
        self.images = self.images.astype(np.float32)
        self.images = np.multiply(self.images, (high - low) / 255.0) + low
        self.num_images = len(self.images)
        self.indices = np.arange(self.num_images, dtype=np.int32)
        if shuffle:
            np.random.shuffle(self.indices)
        if num_images:
            self.num_images = min(self.num_images, num_images)
            self.indices = self.indices[:self.num_images]
        self.attributes = self.attributes[self.indices]
        self.images = self.images[self.indices]
        print(self.images.shape)
        self.data_info = ['{' + '\n'
                          + '\t\'id\': ' + str(i + 1) + '\n'
                          + '\t\'attributes\': ' + str(attr)
                          + '\n}' for i, attr in enumerate(self.attributes)]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def maybe_download(filename, data_dir, SOURCE_URL):
    """Download the data from Yann's website, unless it's already here."""
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


def check_file(data_dir):
    if os.path.exists(data_dir):
        return True
    else:
        os.mkdir(data_dir)
        return False


def uzip_data(decompression_command, decompression_optional, target_path):
    # uzip mnist data
    cmd = [decompression_command, decompression_optional, target_path]
    print('decompress', target_path)
    subprocess.call(cmd)


def cifar10_download(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/'
    filename = 'cifar-10-python.tar.gz'

    if os.path.exists(os.path.join(data_dir, filename)):
        print("[warning...]", filename, "already exist.")
    else:
        target_path = os.path.join(data_dir, filename)
        target_url = os.path.join(SOURCE_URL, filename)
        # maybe_download(filename, data_dir, SOURCE_URL)
        cmd = ['curl', target_url, '-o', target_path]
        print('Downloading CIFAR10')
        subprocess.call(cmd)

    decompressioned_name = 'cifar-10-batches-py'
    if os.path.exists(os.path.join(data_dir, decompressioned_name)):
        print("[warning...]", decompressioned_name, "already exist.")
    else:
        print("data_dir = ", data_dir)
        target_path = os.path.join(data_dir, filename)
        print("target_path = ", target_path)
        tarfile.open(target_path, 'r:gz').extractall(data_dir)


def maybe_download_cifar(train_dir, train=True, one_hot=True):
    cifar10_download(train_dir)
    if train is True:
        images = np.zeros((50000,32,32,3))
        labels = np.zeros((50000,1))
        for j in range(1, 6):
            dataName = os.path.join(train_dir,'cifar-10-batches-py', 'data_batch_' + str(j))
            Xtr = unpickle(dataName)
            for i in range(0, 10000):
                img = np.reshape(Xtr['data'][i], (3, 32, 32))
                img = img.transpose(1, 2, 0)
                images[(j-1)*10000+i] = img
                labels[(j-1)*10000+i] = Xtr['labels'][i]
    else:
        images = np.zeros((10000, 32, 32, 3))
        labels = np.zeros((10000, 1))
        dataName = os.path.join(train_dir, 'cifar-10-batches-py', 'test_batch')
        Xtr = unpickle(dataName)
        for i in range(0, 10000):
            img = np.reshape(Xtr['data'][i], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            images[i] = img
            labels[i] = Xtr['labels'][i]
    return images,labels

if __name__ == '__main__':
    db = CIFARDataSet('./data/cifar', shuffle=True)
    x, y = db[:9]
    saveSampleResults(x, 'test.png', 3, 2)
    print(y)
    print(x.shape, y.shape)
