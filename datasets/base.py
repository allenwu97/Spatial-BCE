import numpy as np
import torch
from PIL import Image
from torch.utils import data


class _BaseDataset(data.Dataset):
    """
    Base dataset class
    """

    def __init__(
        self,
        root,
        train,
        transform,
        fg_path
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.files = []
        self._set_files()
        if self.train:
            self.fg_list = np.load(fg_path, allow_pickle=True).item()

    def _set_files(self):
        """
        Create a file path/image id list.
        """
        raise NotImplementedError()

    def _load_data(self, image_id):
        """
        Load the image and label in numpy.ndarray
        """
        raise NotImplementedError()

    def __getitem__(self, index):
        image_id, image, cls_label = self._load_data(index)
        if self.transform:
            image = self.transform(image)
        if self.train:
            fg = torch.tensor(self.fg_list[image_id])
            return image_id, image, cls_label, fg
        else:
            return image_id, image, cls_label

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
