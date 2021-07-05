import numpy as np
import gzip
from .dataset_base import BaseDataSet
from utils.data_io import maybe_download_file, labels_to_one_hot

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
start_time = None

class MNISTDataSet(BaseDataSet):

  def __init__(self,dataset_path,
               img_width=28, img_height=28, num_images=None, train=True,
               shuffle=False, low=-1, high=1):
    BaseDataSet.__init__(self, dataset_path, img_height, img_width)
    self.images, self.attributes = maybe_download_minst(dataset_path, train)
    self.images = self.images.astype(np.float32)
    self.images = (self.images - 127.5) / 127.5
    self.images = np.expand_dims(self.images, axis=-1)
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

def extract_images(fp):
  with gzip.open(fp, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
  data = data.reshape(-1, 28, 28)
  return data

def extract_labels(fp, one_hot=True):
  with gzip.open(fp, 'rb') as f:
    labels = np.frombuffer(f.read(), np.uint8, offset=8)
  if one_hot:
    labels = labels_to_one_hot(labels, num_classes=10)
  return labels

def maybe_download_minst(train_dir, train=True, one_hot=True):
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = maybe_download_file(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES)
  with open(local_file, 'rb') as f:
    train_images = extract_images(f)

  local_file = maybe_download_file(TRAIN_LABELS, train_dir,
                                   SOURCE_URL + TRAIN_LABELS)
  with open(local_file, 'rb') as f:
    train_labels = extract_labels(f, one_hot=one_hot)

  local_file = maybe_download_file(TEST_IMAGES, train_dir,
                                   SOURCE_URL + TEST_IMAGES)
  with open(local_file, 'rb') as f:
    test_images = extract_images(f)

  local_file = maybe_download_file(TEST_LABELS, train_dir,
                                   SOURCE_URL + TEST_LABELS)
  with open(local_file, 'rb') as f:
    test_labels = extract_labels(f, one_hot=one_hot)

  if train:
    return train_images, train_labels
  else:
    return test_images, test_labels


if __name__ == '__main__':
    maybe_download_minst('./datasets/mnist', train=True, one_hot=True)