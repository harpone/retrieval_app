import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F
import PIL
import io
from PIL import Image
import requests
import warnings
import pickle
from google.cloud import storage
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import imgaug as ia
import skimage.draw
import cv2
import lmdb
import six
import pyarrow as pa
from termcolor import colored
from pyinstrument import Profiler
import webdataset as wds
from webdataset.filters import getfirst, default_collation_fn, batched, shuffle

from core.dataio import image_from_url


class URLDataset(Dataset):
    """Generic dataset for any jpeg URLs.

    """

    def __init__(self,
                 url_list=None,
                 transform=None):
        """Custom PyTorch dataset for a list/array of URLs.

        :param url_list: list or numpy array of dtype np.string_ of `path`s where path is of form 'https://.../xyz.jpg'
        
        :param transform:
        :param
        """

        self.url_list = url_list
        self.transform = transform

    def __len__(self):
        return len(self.url_list)

    def __getitem__(self, index):

        url = str(self.url_list[index], encoding='utf-8')  # e.g. 'https://*.jpg'

        img = image_from_url(url)

        if img is None:
            return None, None, None  # let collate_fn handle missing images

        shape_orig = np.array(img.size)

        if self.transform is not None:
            img = self.transform(img)

        return img, url, shape_orig

