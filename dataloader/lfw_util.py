import os
import random
import numpy as np
import sys
from PIL import Image
import pickle
from .dataset_base import BaseDataSet
import h5py

class LFWDataSet(BaseDataSet):
    def __init__(self, dataset_path, cache=None, img_width=128, img_height=128, num_images=None, train=True,
                 shuffle=False, low=-1, high=1):
        super.__init__(self, dataset_path, img_height, img_width)
        self.dataset_path = dataset_path
        self.img_height = img_height
        self.img_width = img_width

        lfw_att_73 = pickle.load(open(os.path.join(dataset_path, 'lfw_att_73.pickle'), 'rb'))
        self.attr_name = lfw_att_73['AttrName']
        self.img_paths = [name.replace('\\', '/') for name in lfw_att_73['name']]
        self.label = np.asarray(lfw_att_73['label'], dtype=np.float32).transpose()
        if cache != None and os.path.exists(cache):
            self.images = np.asarray(h5py.File(cache, 'r')['images'], dtype=np.float32)
        else:
            print('Preloading images...')
            self.images = np.zeros(shape=(len(self.img_paths), img_width, img_height, 3), dtype=np.float32)
            for i, img in enumerate(self.img_paths):
                image = Image.open(os.path.join(dataset_path, 'lfw', img)).convert('RGB')
                image = image.resize((img_width, img_height))
                image = np.asarray(image, dtype=float)
                cmax = image.max()
                cmin = image.min()
                image = (image - cmin) / (cmax - cmin) * (high - low) + low
                self.images[i] = image
            if cache != None:
                with h5py.File(cache, 'w') as h5:
                    h5.create_dataset('images', data=self.images)
        
        print('Load images of shape {}'.format(self.images.shape))

        indices_train_test = pickle.load(open(os.path.join(dataset_path, 'indices_train_test.pickle'), 'rb'))
        if train:
            self.indices = indices_train_test['indices_img_train'].astype(np.int32).squeeze() - 1
        else:
            self.indices = indices_train_test['indices_img_test'].astype(np.int32).squeeze() - 1
        if not shuffle:
            self.indices.sort()
        self.indices = self.indices[:num_images]
        self.num_images = len(self.indices)
        self.attributes = self.label[self.indices]
        self.images = self.images[self.indices]
        self.data_info = []
        for i in range(self.num_images):
            idx = self.indices[i]
            self.data_info.append('{' + '\n'
                + '\t\"id\": ' + str(idx) + ',\n'
                + '\t\"filename\": \"' + self.img_paths[idx] + '\",\n'
                + '\t\"attributes\": [\n\t\t' + ',\n\t\t'.join(['\"' + self.attr_name[ai] + '\"' for ai, attr in enumerate(self.label[idx]) if attr == 1]) + '\n\t]' + '\n'
            + '}')
        print('Images loaded, shape: {}'.format(self.images.shape))

if __name__ == '__main__':
    #imgs = load_image_paths('../../data/CUB_200_2011/CUB_200_2011')
    #train_idx, test_idx = load_train_test_split('../../data/CUB_200_2011/CUB_200_2011')
    #labels = load_image_attribute_labels('../../data/CUB_200_2011/CUB_200_2011')
    train_data = LFWDataSet('../../data/LFW/LFWA+', num_images=200, shuffle=False, train=True)
    print(train_data.attr_name)
    img, label = train_data[0]
    # attr_names = load_attribute_names('../../data/CUB_200_2011/CUB_200_2011')
    # print(img.shape)
    print(label)
    print(train_data)
