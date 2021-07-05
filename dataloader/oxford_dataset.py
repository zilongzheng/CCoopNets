import os
import h5py
import numpy as np
import scipy.io
from PIL import Image
from utils.data_io import labels_to_one_hot
from .dataset_base import BaseDataSet

class Oxford17DataSet(BaseDataSet):
    def __init__(self, dataset_path, cache=None, img_width=128, img_height=128, num_images=None, train=True,
                 shuffle=False, low=-1, high=1):
        self.dataset_path = dataset_path
        self.img_height = img_height
        self.img_width = img_width
        self.img_paths = sorted([f for f in os.listdir(os.path.join(dataset_path, 'jpg')) if f.endswith('.jpg')])

        if cache != None and os.path.exists(cache):
            with h5py.File(cache, 'r') as h5:
                self.images = np.asarray(h5['images'], dtype=np.float32)
            # self.labels = pickle.load(cache + '.pkl')
        else:
            self.images = np.zeros(shape=(len(self.img_paths), img_width, img_height, 3), dtype=np.float32)
            for i, img in enumerate(self.img_paths):
                # idx = self.indices[i]
                image = Image.open(os.path.join(dataset_path, 'jpg', img)).convert('RGB')
                image = image.resize((img_width, img_height))
                image = np.asarray(image, dtype=np.float32)
                cmax = image.max()
                cmin = image.min()
                image = (image - cmin) / (cmax - cmin) * (high - low) + low
                self.images[i] = image
            if cache != None:
                with h5py.File(cache, 'w') as h5:
                    h5.create_dataset(name='images', data=self.images)
        # print('Load images, shape: {}'.format(self.images.shape))
        self.labels = np.concatenate([[i] * 80 for i in range(17)], axis=0)
        # print(self.labels.shape)
        self.labels = labels_to_one_hot(self.labels, num_classes=17)
        # print(self.labels.shape)
        indices_train_test = scipy.io.loadmat(os.path.join(dataset_path, 'datasplits.mat'))
        if train:
            self.indices = np.concatenate([np.array(indices_train_test[phase], dtype=np.int32).squeeze() - 1 
                    for phase in indices_train_test if phase.startswith('trn') or phase.startswith('val')], axis=0)
        else:
            self.indices = np.concatenate([np.array(indices_train_test[phase], dtype=np.int32).squeeze() - 1 
                    for phase in indices_train_test if phase.startswith('tst')], axis=0)
        if not shuffle:
            self.indices.sort()
        else:
            np.random.shuffle(self.indices)
        self.indices = self.indices[:num_images]
        self.num_images = len(self.indices)
        self.images = self.images[self.indices]
        self.attributes = self.labels[self.indices]
        self.data_info = []
        for i, idx in enumerate(self.indices):
            self.data_info.append('{' + '\n'
                + '\tid: ' + str(i) + '\n'
                + '\tfilename: ' + self.img_paths[idx] + '\n'
                + '\tattributes: ' + str(self.labels[idx]) + '\n'
            + '}')
        print('Images loaded, shape: {}'.format(self.images.shape))

if __name__ == '__main__':
    #imgs = load_image_paths('../../data/CUB_200_2011/CUB_200_2011')
    #train_idx, test_idx = load_train_test_split('../../data/CUB_200_2011/CUB_200_2011')
    #labels = load_image_attribute_labels('../../data/CUB_200_2011/CUB_200_2011')
    datapath = '../../attribute2Image/data/Oxford17/'
    train_data = Oxford17DataSet(datapath, cache=datapath + 'data.h5', num_images=200, shuffle=True, train=True)
    # print(train_data)
    img, label = train_data[0]
    # attr_names = load_attribute_names('../../data/CUB_200_2011/CUB_200_2011')
    # print(img.shape)
    print(label)
    print(train_data)
