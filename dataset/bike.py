from __future__ import print_function, absolute_import
import os
import numpy as np
import random
from .imdb import Imdb
from pathlib import Path
import xml.etree.ElementTree as ET
from evaluate.eval_voc import voc_eval
import cv2

class Bike(Imdb):
    """
    Implementation of Imdb for Pascal VOC datasets

    Parameters:
    ----------
    image_set : str
        set to be used, can be train, val, trainval, test
    year : str
        year of dataset, can be 2007, 2010, 2012...
    devkit_path : str
        devkit path of VOC dataset
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load annotations
    """
    def __init__(self, image_set, base_path, class_names=None, shuffle=True, is_train=True):
        super(Bike, self).__init__('bike_' + image_set)

        self.image_set = image_set
        self.data_path = os.path.join(base_path, self.image_set)
        self.is_train = is_train

        self.classes = class_names.strip().split(',')
        self.num_classes = len(self.classes)

        self.image_set_index = None
        self.labels = None
        self._load_all(shuffle)
        self.num_images = len(self.image_set_index)

    @property
    def cache_path(self):
        """
        make a directory to store all caches

        Returns:
        ---------
            cache path
        """
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = Path(self.data_path) / name
        assert image_file.exists(), 'Path does not exist: {}'.format(image_file)
        return image_file.as_posix()

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def _load_all(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """

        image_set_index = []
        labels = []

        # img_path = sorted(dir_path.glob("*.jpg"))[0]
        dir_path = Path(self.data_path)
        for img_path in dir_path.glob("*.jpg"):
            img_basename = img_path.name.split('.')[0]
            img_xml = img_path.parent / '{}.xml'.format(img_basename)
            assert img_xml.exists() is True

            tree = ET.parse(img_xml)
            root = tree.getroot()
            size = root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)
            folder_name = root.find('folder').text
            label = []

            # obj=[x for x in root.iter('object')][0]
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                cls_id = self.classes.index(cls_name)
                xml_box = obj.find('bndbox')
                xmin = float(xml_box.find('xmin').text) / width
                ymin = float(xml_box.find('ymin').text) / height
                xmax = float(xml_box.find('xmax').text) / width
                ymax = float(xml_box.find('ymax').text) / height
                difficult = 0 if cls_name != 'irrelevant' else 1
                label.append([cls_id, xmin, ymin, xmax, ymax, difficult])

            if label:
                labels.append(np.array(label))
                image_set_index.append(img_path.name)
                # image_set_index.append(img_path.as_posix())

        if shuffle:
            indices = list(range(len(image_set_index)))
            random.shuffle(indices)
            image_set_index = [image_set_index[i] for i in indices]
            labels = [labels[i] for i in indices]

        # store the results
        self.image_set_index = image_set_index
        self.labels = labels
