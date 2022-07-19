import torch
import torchvision
from .voc12 import VOCAug
from .coco import COCOAug


def get_dataset(dataset, data_dir, transform, train=True, fg_path=None):
    if dataset == 'voc12':
        dataset = VOCAug(root=data_dir, train=train, transform=transform, fg_path=fg_path)
    elif dataset == 'coco':
        dataset = COCOAug(root=data_dir, train=train, transform=transform, fg_path=fg_path)
    else:
        raise NotImplementedError
    return dataset