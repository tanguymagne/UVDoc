import json
import math
import os
import warnings
from os.path import join as pjoin

import cv2
import h5py as h5
import numpy as np
import torch

from data_utils import BaseDataset, get_geometric_transform
from utils import GRID_SIZE, IMG_SIZE, bilinear_unwarping


class UVDocDataset(BaseDataset):
    """
    Torch dataset class for the UVDoc dataset.
    """

    def __init__(
        self,
        data_path="./data/UVdoc",
        appearance_augmentation=[],
        geometric_augmentations=[],
        grid_size=GRID_SIZE,
    ) -> None:
        super().__init__(
            data_path=data_path,
            appearance_augmentation=appearance_augmentation,
            img_size=IMG_SIZE,
            grid_size=grid_size,
        )
        self.original_grid_size = (89, 61)  # size of the captured data
        self.grid3d_normalization = (0.11433014, -0.12551452, 0.12401487, -0.12401487, 0.1952378, -0.1952378)
        self.geometric_transform = get_geometric_transform(geometric_augmentations, gridsize=self.original_grid_size)

        self.all_samples = [x[:-4] for x in os.listdir(pjoin(self.dataroot, "img")) if x.endswith(".png")]

    def __getitem__(self, index):
        # Get all paths
        sample_id = self.all_samples[index]
        with open(pjoin(self.dataroot, "metadata_sample", f"{sample_id}.json"), "r") as f:
            sample_name = json.load(f)["geom_name"]
        img_path = pjoin(self.dataroot, "img", f"{sample_id}.png")
        grid2D_path = pjoin(self.dataroot, "grid2d", f"{sample_name}.mat")
        grid3D_path = pjoin(self.dataroot, "grid3d", f"{sample_name}.mat")

        # Load 2D grid, 3D grid and image. Normalize 3D grid
        with h5.File(grid2D_path, "r") as file:
            grid2D_ = np.array(file["grid2d"][:].T.transpose(2, 0, 1))  # scale in range of img resolution

        with h5.File(grid3D_path, "r") as file:
            grid3D = np.array(file["grid3d"][:].T)

        if self.normalize_3Dgrid:  # scale grid3D to [0,1], based on stats computed over the entire dataset
            xmx, xmn, ymx, ymn, zmx, zmn = self.grid3d_normalization
            grid3D[:, :, 0] = (grid3D[:, :, 0] - xmn) / (xmx - xmn)
            grid3D[:, :, 1] = (grid3D[:, :, 1] - ymn) / (ymx - ymn)
            grid3D[:, :, 2] = (grid3D[:, :, 2] - zmn) / (zmx - zmn)
            grid3D = np.array(grid3D, dtype=np.float32)
        grid3D = torch.from_numpy(grid3D.transpose(2, 0, 1))

        img_RGB_ = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # Pixel-wise augmentation
        img_RGB_ = self.appearance_transform(image=img_RGB_)["image"]

        # Geometric Augmentations
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            transformed = self.geometric_transform(
                image=img_RGB_,
                keypoints=grid2D_.transpose(1, 2, 0).reshape(-1, 2),
            )
            img_RGB_ = transformed["image"]

            grid2D_ = np.array(transformed["keypoints"]).reshape(*self.original_grid_size, 2).transpose(2, 0, 1)

            flipped = False
            for x in transformed["replay"]["transforms"]:
                if "SafeHorizontalFlip" in x["__class_fullname__"]:
                    flipped = x["applied"]
            if flipped:
                grid3D[1] = 1 - grid3D[1]
                grid3D = torch.flip(grid3D, dims=(2,))

        # Tight crop
        grid2Dtmp = grid2D_
        img_RGB, grid2D = self.crop_tight(img_RGB_, grid2Dtmp)

        # Subsample grids to desired resolution
        row_sampling_factor = math.ceil(self.original_grid_size[0] / self.grid_size[0])
        col_sampling_factor = math.ceil(self.original_grid_size[1] / self.grid_size[1])
        grid3D = grid3D[:, ::row_sampling_factor, ::col_sampling_factor]
        grid2D = grid2D[:, ::row_sampling_factor, ::col_sampling_factor]
        grid2D = torch.from_numpy(grid2D).float()

        # Unwarp the image according to grid
        img_RGB_unwarped = bilinear_unwarping(img_RGB.unsqueeze(0), grid2D.unsqueeze(0), self.img_size).squeeze()

        return (
            img_RGB.float() / 255.0,
            img_RGB_unwarped.float() / 255.0,
            grid2D,
            grid3D,
        )
