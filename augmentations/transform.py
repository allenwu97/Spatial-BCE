import torchvision.transforms as T
from randaugment import RandAugment
import PIL
import numpy as np
import torch

imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


class Transform():
    def __init__(self, image_size, mean_std=imagenet_mean_std, train=True):
        image_size = 321 if image_size is None else image_size
        self.pre_processing = T.Compose([T.Resize(size=(image_size, image_size))])
        self.aug = T.Compose([RandAugment()])
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
        self.train = train
        self.scales = [1, 0.5, 1.5, 2.0]


    def __call__(self, image):
        if self.train:
            image = self.pre_processing(image)
            image = self.aug(image)
            image = self.transform(image)
            return image
        else:
            ms_img_list = []
            for s in self.scales:
                target_size = (round(image.size[0] * s), round(image.size[1] * s))
                s_img = image.resize(target_size, resample=PIL.Image.CUBIC)
                ms_img_list.append(s_img)

            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

            msf_img_list = []
            for i in range(len(ms_img_list)):
                msf_img_list.append(ms_img_list[i])
                msf_img_list.append(torch.flip(ms_img_list[i], [2]).clone())
            return msf_img_list












