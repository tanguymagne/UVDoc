import random

import albumentations as A
import cv2
import numpy as np
import torch

from data_custom_augmentations import SafeHorizontalFlip, SafePerspective
from utils import GRID_SIZE, IMG_SIZE


def get_appearance_transform(transform_types):
    """
    Returns an albumentation compose augmentation.

    transform_type is a list containing types of pixel-wise data augmentation to use.
    Possible augmentations are 'shadow', 'blur', 'visual', 'noise', 'color'.
    """

    transforms = []
    if "shadow" in transform_types:
        transforms.append(A.RandomShadow(p=0.1))
    if "blur" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.Defocus(p=5),
                    A.Downscale(p=15, interpolation=cv2.INTER_LINEAR),
                    A.GaussianBlur(p=65),
                    A.MedianBlur(p=15),
                ],
                p=0.75,
            )
        )
    if "visual" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.ToSepia(p=15),
                    A.ToGray(p=20),
                    A.Equalize(p=15),
                    A.Sharpen(p=20),
                ],
                p=0.5,
            )
        )
    if "noise" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.GaussNoise(var_limit=(10.0, 20.0), p=70),
                    A.ISONoise(intensity=(0.1, 0.25), p=30),
                ],
                p=0.6,
            )
        )
    if "color" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.ColorJitter(p=5),
                    A.HueSaturationValue(p=10),
                    A.RandomBrightnessContrast(brightness_limit=[-0.05, 0.25], p=85),
                ],
                p=0.95,
            )
        )

    return A.Compose(transforms=transforms)


def get_geometric_transform(transform_types, gridsize):
    """
    Returns an albumentation compose augmentation.

    transform_type is a list containing types of geometric data augmentation to use.
    Possible augmentations are 'rotate', 'flip' and 'perspective'.
    """

    transforms = []
    if "rotate" in transform_types:
        transforms.append(
            A.SafeRotate(
                limit=[-30, 30],
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.5,
            )
        )
    if "flip" in transform_types:
        transforms.append(SafeHorizontalFlip(gridsize=gridsize, p=0.25))

    if "perspective" in transform_types:
        transforms.append(SafePerspective(p=0.5))

    return A.ReplayCompose(
        transforms=transforms,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def crop_image_tight(img, grid2D):
    """
    Crops the image tightly around the keypoints in grid2D.
    This function creates a tight crop around the document in the image.
    """
    size = img.shape

    minx = np.floor(np.amin(grid2D[0, :, :])).astype(int)
    maxx = np.ceil(np.amax(grid2D[0, :, :])).astype(int)
    miny = np.floor(np.amin(grid2D[1, :, :])).astype(int)
    maxy = np.ceil(np.amax(grid2D[1, :, :])).astype(int)
    s = 20
    s = min(min(s, minx), miny)  # s shouldn't be smaller than actually available natural padding is
    s = min(min(s, size[1] - 1 - maxx), size[0] - 1 - maxy)

    # Crop the image slightly larger than necessary
    img = img[miny - s : maxy + s, minx - s : maxx + s, :]
    cx1 = random.randint(0, max(s - 5, 1))
    cx2 = random.randint(0, max(s - 5, 1)) + 1
    cy1 = random.randint(0, max(s - 5, 1))
    cy2 = random.randint(0, max(s - 5, 1)) + 1

    img = img[cy1:-cy2, cx1:-cx2, :]
    top = miny - s + cy1
    bot = size[0] - maxy - s + cy2
    left = minx - s + cx1
    right = size[1] - maxx - s + cx2
    return img, top, bot, left, right


class BaseDataset(torch.utils.data.Dataset):
    """
    Base torch dataset class for all unwarping dataset.
    """

    def __init__(
        self,
        data_path,
        appearance_augmentation=[],
        img_size=IMG_SIZE,
        grid_size=GRID_SIZE,
    ) -> None:
        super().__init__()

        self.dataroot = data_path
        self.img_size = img_size
        self.grid_size = grid_size
        self.normalize_3Dgrid = True

        self.appearance_transform = get_appearance_transform(appearance_augmentation)

        self.all_samples = []

    def __len__(self):
        return len(self.all_samples)

    def crop_tight(self, img_RGB, grid2D):
        # The incoming grid2D array is expressed in pixel coordinates (resolution of img_RGB before crop/resize)
        size = img_RGB.shape
        img, top, bot, left, right = crop_image_tight(img_RGB, grid2D)
        img = cv2.resize(img, self.img_size)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        grid2D[0, :, :] = (grid2D[0, :, :] - left) / (size[1] - left - right)
        grid2D[1, :, :] = (grid2D[1, :, :] - top) / (size[0] - top - bot)
        grid2D = (grid2D * 2.0) - 1.0

        return img, grid2D
