from .transform import Transform


imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


def get_aug(image_size, train):
    augmentation = Transform(image_size, mean_std=imagenet_mean_std, train=train)

    return augmentation








