import os
import numpy as np
from PIL import Image
import pandas as pd
import h5py
from tqdm import tqdm

class CelebADataSet(object):
    def __init__(self, dataset_path, cache=None, img_width=128, img_height=128, num_images=None, train=True,
                 shuffle=False, low=-1, high=1, crop='center'):
        self.dataset_path = dataset_path
        self.img_height = img_height
        self.img_width = img_width
        self.filename_attr, self.attr_names = load_list_attr(self.dataset_path)
        self.filename_bbox = load_list_bbox(self.dataset_path)
        self.filenames = self.filename_attr.keys()
        if cache != None and os.path.exists(cache):
            with h5py.File(cache, 'r') as h5:
                self.images = np.asarray(h5['images'], dtype=np.float32)
        else:
            print('Preloading images...')
            self.images = []
            for i, fn in tqdm(enumerate(self.filenames)):
                image = Image.open(os.path.join(self.dataset_path, 'Img', 'img_align_celeba', fn)).convert('RGB')
                if crop == 'center':
                    im_w, im_h = image.size
                    x_top = (im_w - img_width) // 2 if im_w > img_width else 0
                    y_top = (im_h - img_height) // 2 if im_h > img_height else 0
                    right = min(x_top+img_width, im_w)
                    lower = min(y_top+img_height, im_h)
                    image = image.crop((x_top, y_top, right, lower))
                elif crop == 'bbox':
                    x, y, w, h = self.filename_bbox[fn]
                    image = image.crop((x, y, x+w, y+h))
                else:
                    raise NotImplementedError
                
                image = image.resize((img_width, img_height))
                image = np.asarray(image, dtype=np.float32)
                cmax = image.max()
                cmin = image.min()
                image = (image - cmin) / (cmax - cmin) * (high - low) + low
                self.images.append(image)
            self.images = np.array(self.images, dtype=np.float32)

            if cache != None:
                with h5py.File(cache, 'w') as h5:
                    h5.create_dataset('images', data=self.images)
                
        print('Images loaded, shape: {}'.format(self.images.shape))
        self.ids = np.arange(len(self.filenames))
        if shuffle:
            np.random.shuffle(self.ids)
        
        self.num_images = len(self.filenames)
        if num_images != None:
            self.num_images = min(self.num_images, num_images)
        
        self.data_info = []
        
        for i in range(self.num_images):
            img_id = self.ids[i]
            fn = self.filenames[img_id]
            self.data_info.append('{' + '\n'
                + '\t\'id\': ' + str(i+1) + ',\n'
                + '\t\'filename\': \'' + fn + '\',\n'
                + '\t\'attributes\': [\n\t\t'
                                  + ',\n\t\t'.join(['\'' + an + '\'' for ia, an in enumerate(self.attr_names) \
                                                    if self.filename_attr[fn][ia] == 1])
                                  + '\n\t]' + '\n'
            + '}')


    def __getitem__(self, index):
        attr = np.asarray([self.filename_attr[self.filenames[id]] for id in self.ids[index]], dtype=np.float32)
        return (self.images[self.ids[index]], attr)

    def __len__(self):
        return self.num_images

    def __str__(self):
        return ', '.join(self.data_info)

def load_list_bbox(dataset_path=''):
    bbox_path = os.path.join(dataset_path, 'Anno', 'list_bbox_celeba.txt')
    df_bbox = pd.read_csv(bbox_path, delim_whitespace=True, header=1)
    bbox = {}
    for i, row in df_bbox.iterrows():
        bbox[row['image_id']] = df_bbox.iloc[i][1:].astype(int).tolist()
    return bbox

def load_list_attr(dataset_path=''):
    attr_path = os.path.join(dataset_path, 'Anno', 'list_attr_celeba.txt')
    df_attr = pd.read_csv(attr_path, delim_whitespace=True, header=1).astype(int)
    filename_attr = {}
    for fn, row in df_attr.iterrows():
        filename_attr[fn] = row.tolist()
    return filename_attr, list(df_attr)