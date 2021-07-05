import os
import random
import numpy as np
from PIL import Image
from .dataset_base import BaseDataSet
from collections import Counter


class CUBDataSet(BaseDataSet):
    def __init__(self, dataset_path, img_width=128, img_height=128, num_images=None, train=True,
                 shuffle=False, low=-1, high=1):
        super.__init__(self, dataset_path, img_height, img_width)
        self.dataset_path = dataset_path
        self.img_height = img_height
        self.img_width = img_width
        self.img_paths = load_image_paths(dataset_path, path_prefix='images')
        self.train_id, self.test_id = load_train_test_split(dataset_path)
        self.img_attr, self.img_info = load_image_attribute_labels(dataset_path)
        self.image_bboxes = load_bounding_box_annotations(dataset_path)
        self.attr_names = load_attribute_names(dataset_path)
        if train:
            self.id = self.train_id
        else:
            self.id = self.test_id
        if shuffle:
            np.random.shuffle(self.id)
        else:
            self.id.sort()
        if num_images:
            self.num_images = min(num_images, len(self.id))
        else:
            self.num_images = len(self.id)
        self.images = np.zeros(shape=(self.num_images, img_width, img_height, 3), dtype=np.float32)
        self.attributes = np.zeros(shape=(self.num_images, 312), dtype=np.float32)
        self.data_info = []
        for i in range(self.num_images):
            img_id = self.id[i]
            image = Image.open(os.path.join(dataset_path, self.img_paths[img_id])).convert('RGB')
            x, y, w, h = self.image_bboxes[img_id]
            image = image.crop((x, y, x+w, y+h))
            image = image.resize((img_width, img_height))
            image = np.asarray(image, dtype=float)
            cmax = image.max()
            cmin = image.min()
            image = (image - cmin) / (cmax - cmin) * (high - low) + low
            self.images[i] = image
            self.attributes[i] = self.img_attr[img_id]
            self.data_info.append('{' + '\n'
                + '\t\'id\': ' + str(i+1) + '\n'
                + '\t\'filename\': \'' + self.img_paths[img_id] + '\'\n'
                + '\t\'attributes\': [\n\t\t'
                                  + ',\n\t\t'.join(['\'' + self.attr_names[str(id+1)] + '\'' for id in range(len(self.attributes[i]))
                                                    if self.attributes[i][id] == 1])
                                  + '\n\t]' + '\n'
            + '}')
        print('Images loaded, shape: {}'.format(self.images.shape))

def format_labels(image_labels):
    """
    Convert the image labels to be integers between [0, num classes)

    Returns :
      condensed_image_labels = { image_id : new_label}
      new_id_to_original_id_map = {new_label : original_label}
    """

    label_values = list(set(image_labels.values()))
    label_values.sort()
    condensed_image_labels = dict([(image_id, label_values.index(label))
                                   for image_id, label in image_labels.iteritems()])
    new_id_to_original_id_map = dict([[label_values.index(label), label] for label in label_values])

    return condensed_image_labels, new_id_to_original_id_map


def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = int(pieces[0])
            names[class_id] = ' '.join(pieces[1:])

    return names


def load_image_labels(dataset_path=''):
    labels = {}

    with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            class_id = pieces[1]
            labels[image_id] = int(class_id)  # GVH: should we force this to be an int?

    return labels


def load_image_paths(dataset_path='', path_prefix=''):
    paths = {}

    with open(os.path.join(dataset_path, 'images.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            path = os.path.join(path_prefix, pieces[1])
            paths[image_id] = path

    return paths

def load_attribute_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'attributes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            attr_id = pieces[0]
            name = pieces[1]
            names[attr_id] = name

    return names



def load_image_attribute_labels(dataset_path=''):
    labels = {}
    info = {}
    attr_names = load_attribute_names(dataset_path)
    with open(os.path.join(dataset_path, 'attributes/image_attribute_labels.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            attr_id = int(pieces[1]) - 1
            is_present = pieces[2]
            certainties = int(pieces[3])
            if image_id not in labels:
                labels[image_id] = np.zeros(312, dtype=np.uint8)
                info[image_id] = []
            if is_present == '1' and certainties >= 4:
                labels[image_id][attr_id] = 1
                info[image_id].append(attr_names[pieces[1]])
    return labels, info

def load_bounding_box_annotations(dataset_path=''):
    bboxes = {}

    with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            bbox = map(int, map(float, pieces[1:]))
            bboxes[image_id] = bbox

    return bboxes


def load_part_annotations(dataset_path=''):
    parts_d = {}

    with open(os.path.join(dataset_path, 'parts/part_locs.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            parts_d.setdefault(image_id, {})
            part_id = int(pieces[1])
            parts_d[image_id][part_id] = map(float, pieces[2:])

    # convert the dictionary to an array
    parts = {}
    for image_id, parts_dict in parts_d.items():
        keys = parts_dict.keys()
        keys.sort()
        parts_list = []
        for part_id in keys:
            parts_list += parts_dict[part_id]
        parts[image_id] = parts_list

    return parts


def load_train_test_split(dataset_path=''):
    train_images = []
    test_images = []

    with open(os.path.join(dataset_path, 'train_test_split.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            is_train = int(pieces[1])
            if is_train > 0:
                train_images.append(image_id)
            else:
                test_images.append(image_id)

    return np.asarray(train_images), np.asarray(test_images)


def load_image_sizes(dataset_path=''):
    sizes = {}

    with open(os.path.join(dataset_path, 'sizes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            width, height = map(int, pieces[1:])
            sizes[image_id] = [width, height]

    return sizes


# Not the best python code etiquette, but trying to keep everything self contained...
def create_image_sizes_file(dataset_path, image_path_prefix):
    from scipy.misc import imread

    image_paths = load_image_paths(dataset_path, image_path_prefix)
    image_sizes = []
    for image_id, image_path in image_paths.iteritems():
        im = imread(image_path)
        image_sizes.append([image_id, im.shape[1], im.shape[0]])

    with open(os.path.join(dataset_path, 'sizes.txt'), 'w') as f:
        for image_id, w, h in image_sizes:
            f.write("%s %d %d\n" % (str(image_id), w, h))


def format_dataset(dataset_path, image_path_prefix):
    """
    Load in a dataset (that has been saved in the CUB Format) and store in a format
    to be written to the tfrecords file
    """

    image_paths = load_image_paths(dataset_path, image_path_prefix)
    image_sizes = load_image_sizes(dataset_path)
    image_bboxes = load_bounding_box_annotations(dataset_path)
    image_parts = load_part_annotations(dataset_path)
    image_labels, new_label_to_original_label_map = format_labels(load_image_labels(dataset_path))
    class_names = load_class_names(dataset_path)
    train_images, test_images = load_train_test_split(dataset_path)

    train_data = []
    test_data = []

    for image_ids, data_store in [(train_images, train_data), (test_images, test_data)]:
        for image_id in image_ids:

            width, height = image_sizes[image_id]
            width = float(width)
            height = float(height)

            x, y, w, h = image_bboxes[image_id]
            x1 = max(x / width, 0.)
            x2 = min((x + w) / width, 1.)
            y1 = max(y / height, 0.)
            y2 = min((y + h) / height, 1.)

            parts_x = []
            parts_y = []
            parts_v = []
            parts = image_parts[image_id]
            for part_index in range(0, len(parts), 3):
                parts_x.append(max(parts[part_index] / width, 0.))
                parts_y.append(max(parts[part_index + 1] / height, 0.))
                parts_v.append(int(parts[part_index + 2]))

            data_store.append({
                "filename": image_paths[image_id],
                "id": image_id,
                "class": {
                    "label": image_labels[image_id],
                    "text": class_names[new_label_to_original_label_map[image_labels[image_id]]]
                },
                "object": {
                    "count": 1,
                    "bbox": {
                        "xmin": [x1],
                        "xmax": [x2],
                        "ymin": [y1],
                        "ymax": [y2],
                        "label": [image_labels[image_id]],
                        "text": [class_names[new_label_to_original_label_map[image_labels[image_id]]]]
                    },
                    "parts": {
                        "x": parts_x,
                        "y": parts_y,
                        "v": parts_v
                    },
                    "id": [image_id],
                    "area": [w * h]
                }
            })

    return train_data, test_data


def create_validation_split(train_data, fraction_per_class=0.1, shuffle=True):
    """
    Take `images_per_class` from the train dataset and create a validation set.
    """

    subset_train_data = []
    val_data = []
    val_label_counts = {}

    class_labels = [i['class']['label'] for i in train_data]
    images_per_class = Counter(class_labels)
    val_images_per_class = {label: 0 for label in images_per_class.keys()}

    # Sanity check to make sure each class has more than 1 label
    for label, image_count in images_per_class.items():
        if image_count <= 1:
            print("Warning: label %d has only %d images" % (label, image_count))

    if shuffle:
        random.shuffle(train_data)

    for image_data in train_data:
        label = image_data['class']['label']

        if label not in val_label_counts:
            val_label_counts[label] = 0

        if val_images_per_class[label] < images_per_class[label] * fraction_per_class:
            val_data.append(image_data)
            val_images_per_class[label] += 1
        else:
            subset_train_data.append(image_data)

    return subset_train_data, val_data


if __name__ == '__main__':
    #imgs = load_image_paths('../../data/CUB_200_2011/CUB_200_2011')
    #train_idx, test_idx = load_train_test_split('../../data/CUB_200_2011/CUB_200_2011')
    #labels = load_image_attribute_labels('../../data/CUB_200_2011/CUB_200_2011')
    np.random.seed(1)
    train_data = CUBDataSet('../../data/CUB_200_2011/CUB_200_2011', num_images=1000, shuffle=True, train=True)
    # saveSampleImages(train_data.images, 'obs_img.png', col_num=10, row_num=10, save_all=True)
    attr_names = load_attribute_names('../../data/CUB_200_2011/CUB_200_2011')
    img, attr = train_data[0:50]

    img_attr = [attr_names[str(i+1)] for i, _ in enumerate(attr[14]) if attr[14][i] == 1]
    print(img.shape)
    print(attr.shape)
    print(',\n'.join(img_attr))
