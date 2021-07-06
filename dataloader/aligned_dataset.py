import os
import numpy as np
import h5py
from PIL import Image
from numpy.core.shape_base import stack
from utils.data_io import numpy2image, image2numpy

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def read_images_from_folder(data_path, img_size=224, img_nc=3):
    img_list = sorted([f for f in os.listdir(data_path) if any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS)])
    images = []
    print('Reading images from: {}'.format(data_path))
    for i in range(len(img_list)):
        image = Image.open(os.path.join(data_path, img_list[i]))
        if img_nc == 1:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        image = image.resize((img_size, img_size))
        image = image2numpy(image)
        images.append(image)
    images = np.stack(images, axis=0)
    print('Images loaded, shape: {}'.format(images.shape))
    return images


class AlignedDataSet(object):
    def __init__(self, src_img_path, tgt_img_path, cache=None, load_size=128, crop_size=128, img_nc=3, num_images=None, random_crop=False,
                 random_flip=False, shuffle=False):
        self.src_img_path = src_img_path
        self.tgt_img_path = tgt_img_path
        self.load_size = load_size
        self.crop_size = crop_size
        self.img_size = self.crop_size if random_crop else self.load_size
        self.img_nc = img_nc
        self.random_flip = random_flip
        self.random_crop = random_crop
        if cache != None and os.path.exists(cache):
            with h5py.File(cache, 'r') as h5:
                self.src_images = np.asarray(h5['src_images'], dtype=np.float32)
                self.tgt_images = np.asarray(h5['tgt_images'], dtype=np.float32)
        else:
            self.src_images = read_images_from_folder(src_img_path, load_size, img_nc)
            self.tgt_images = read_images_from_folder(tgt_img_path, load_size, img_nc)
            if cache != None:
                with h5py.File(cache, 'w') as h5:
                    h5.create_dataset('src_images', data=self.src_images)
                    h5.create_dataset('tgt_images', data=self.tgt_images)
        assert len(self.src_images) == len(self.tgt_images), 'number of source and target images must be equal.'
        if num_images:
            self.num_images = min(num_images, len(self.src_images))
        else:
            self.num_images = len(self.src_images)

        if shuffle:
            self.indices = np.random.permutation(self.num_images)
            self.src_images = self.src_images[self.indices]
            self.tgt_images = self.tgt_images[self.indices]

    def __getitem__(self, index):
        src_img = self.src_images[index]
        tgt_img = self.tgt_images[index]

        out_src_img = []
        out_tgt_img = []

        for (src, tgt) in zip(src_img, tgt_img):


            if self.random_flip and np.random.rand() < 0.5:
                # flip array
                src = np.fliplr(src)
                tgt = np.fliplr(tgt)

            if self.random_crop and self.crop_size < self.load_size:
                crop_x = np.random.randint(0, self.load_size - self.crop_size + 1)
                crop_y = np.random.randint(0, self.load_size - self.crop_size + 1)
                src = src[crop_x:crop_x+self.crop_size, crop_y:crop_y+self.crop_size]
                tgt = tgt[crop_x:crop_x+self.crop_size, crop_y:crop_y+self.crop_size]

            out_src_img.append(src)
            out_tgt_img.append(tgt)

        

        return (np.stack(out_src_img, axis=0), np.stack(out_tgt_img, axis=0))

    def __len__(self):
        return self.num_images

