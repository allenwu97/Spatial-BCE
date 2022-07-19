from __future__ import absolute_import, print_function

import os.path as osp
import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from .base import _BaseDataset

class VOCAug(_BaseDataset):
    """
    PASCAL VOC Segmentation dataset with extra annotations
    """

    def __init__(self, **kwargs):
        super(VOCAug, self).__init__(**kwargs)

    def _set_files(self):

        file_list = osp.join('./datasets', 'train_aug' + ".txt")
        file_list = tuple(open(file_list, "r"))
        file_list = [id_.rstrip().split(" ") for id_ in file_list]
        self.files, self.labels = list(zip(*file_list))
        self.label_list = np.load('./datasets/cls_labels.npy', allow_pickle=True).item()


    def _load_data(self, index):
        # Set paths
        image_id = self.files[index].split("/")[-1].split(".")[0]
        image_path = osp.join(self.root, self.files[index][-15:])
        image = Image.open(image_path).convert("RGB")
        label = torch.from_numpy(self.label_list[image_id])

        return image_id, image, label
