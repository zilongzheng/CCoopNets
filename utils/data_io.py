import os
import math
import time
import numpy as np
import sys
from PIL import Image
try:
    from urllib.request import urlretrieve

except ImportError:
    from urllib import urlretrieve
try:
    from urllib.parse import urljoin
except ImportError:
    from urlparse import urljoin

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
start_time = None

def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def maybe_download_file(fname, target_dir, url):
    filepath = os.path.join(target_dir, fname)
    if os.path.exists(filepath):
        print('File {} exists!'.format(filepath))
        return filepath

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    filepath, _ = urlretrieve(url, filepath, reporthook)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', fname, statinfo.st_size, 'bytes.')

    return filepath

def labels_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def cell2img(cell_image, image_size=100, margin_syn=2):
    num_cols = cell_image.shape[1] // image_size
    num_rows = cell_image.shape[0] // image_size
    images = np.zeros((num_cols * num_rows, image_size, image_size, 3))
    for ir in range(num_rows):
        for ic in range(num_cols):
            temp = cell_image[ir*(image_size+margin_syn):image_size + ir*(image_size+margin_syn),
                   ic*(image_size+margin_syn):image_size + ic*(image_size+margin_syn),:]
            images[ir*num_cols+ic] = temp
    return images

def clip_by_value(input_, min=0, max=1):
    return np.minimum(max, np.maximum(min, input_))

def img2cell(images, row_num=10, col_num=10, low=-1, high=1, margin_syn=2):
    [num_images, image_size] = images.shape[0:2]
    num_cells = int(math.ceil(num_images / (col_num * row_num)))
    cell_image = np.zeros((num_cells, row_num * image_size + (row_num-1)*margin_syn,
                           col_num * image_size + (col_num-1)*margin_syn, images.shape[3])).astype(np.uint8)
    for i in range(num_images):
        cell_id = int(math.floor(i / (col_num * row_num)))
        idx = i % (col_num * row_num)
        ir = int(math.floor(idx / col_num))
        ic = idx % col_num
        temp = np.clip(images[i], low, high)
        temp = (temp + 1.) * 127.5
        cell_image[cell_id, (image_size+margin_syn)*ir:image_size + (image_size+margin_syn)*ir,
                    (image_size+margin_syn)*ic:image_size + (image_size+margin_syn)*ic,:] = temp.astype(np.uint8)
    if images.shape[3] == 1:
        cell_image = np.squeeze(cell_image, axis=3)
    return cell_image

def saveSampleImages(sample_results, filename, row_num=10, col_num=10, margin_syn=2, save_all=False):
    cell_image = img2cell(sample_results, row_num=row_num, col_num=col_num, margin_syn=margin_syn)
    if save_all:
        save_dir = filename[:-4]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for ci in range(len(cell_image)):
            Image.fromarray(cell_image[ci]).save(save_dir + '/' + filename[:-4] + '_%03d.png' % ci)
    else:
        Image.fromarray(cell_image[0]).save(filename)