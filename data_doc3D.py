from os.path import join as pjoin

import cv2
import h5py as h5
import numpy as np
import torch

from data_utils import BaseDataset
from utils import GRID_SIZE, IMG_SIZE, bilinear_unwarping


class doc3DDataset(BaseDataset):
    """
    Torch dataset class for the Doc3D dataset.
    """

    def __init__(
        self,
        data_path="./data/doc3D",
        split="train",
        appearance_augmentation=[],
        grid_size=GRID_SIZE,
    ):
        super().__init__(
            data_path=data_path,
            appearance_augmentation=appearance_augmentation,
            img_size=IMG_SIZE,
            grid_size=grid_size,
        )
        self.grid3d_normalization = (1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497)

        if split == "train":
            path = pjoin(self.dataroot, "traindoc.txt")
        elif split == "val":
            path = pjoin(self.dataroot, "valdoc3D.txt")

        with open(path, "r") as files:
            file_list = tuple(files)
        self.all_samples = np.array([id_.rstrip() for id_ in file_list], dtype=np.string_)

    def __getitem__(self, index):
        # Get all paths
        im_name = self.all_samples[index].decode("UTF-8")
        img_path = pjoin(self.dataroot, "img", im_name + ".png")
        grid2D_path = pjoin(self.dataroot, "grid2D", im_name + ".mat")
        grid3D_path = pjoin(self.dataroot, "grid3D", im_name + ".mat")
        bm_path = pjoin(self.dataroot, "bm", im_name + ".mat")

        # Load 2D grid, 3D grid and image. Normalize 3D grid
        with h5.File(grid2D_path, "r") as file:
            grid2D_ = np.array(file["grid2D"][:].T.transpose(2, 0, 1))  # scale in range of img resolution

        with h5.File(grid3D_path, "r") as file:
            grid3D = np.array(file["grid3D"][:].T)

        if self.normalize_3Dgrid:  # scale grid3D to [0,1], based on stats computed over the entire dataset
            xmx, xmn, ymx, ymn, zmx, zmn = self.grid3d_normalization
            grid3D[:, :, 0] = (grid3D[:, :, 0] - zmn) / (zmx - zmn)
            grid3D[:, :, 1] = (grid3D[:, :, 1] - ymn) / (ymx - ymn)
            grid3D[:, :, 2] = (grid3D[:, :, 2] - xmn) / (xmx - xmn)
            grid3D = np.array(grid3D, dtype=np.float32)
        grid3D[:, :, 1] = grid3D[:, :, 1][:, ::-1]
        grid3D[:, :, 1] = 1 - grid3D[:, :, 1]
        grid3D = torch.from_numpy(grid3D.transpose(2, 0, 1))

        img_RGB_ = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Pixel-wise augmentation
        img_RGB_ = self.appearance_transform(image=img_RGB_)["image"]

        # Create unwarped image according to the backward mapping (first load the backward mapping)
        with h5.File(bm_path, "r") as file:
            bm = np.array(file["bm"][:].T.transpose(2, 0, 1))
        bm = ((bm / 448) - 0.5) * 2.0
        bm = torch.from_numpy(bm).float()

        img_RGB_unwarped = bilinear_unwarping(
            torch.from_numpy(img_RGB_.transpose(2, 0, 1)).float().unsqueeze(0),
            bm.unsqueeze(0),
            self.img_size,
        ).squeeze()

        # Tight crop
        grid2Dtmp = grid2D_
        img_RGB, grid2D = self.crop_tight(img_RGB_, grid2Dtmp)

        # Convert 2D grid to torch tensor
        grid2D = torch.from_numpy(grid2D).float()

        return (
            img_RGB.float() / 255.0,
            img_RGB_unwarped.float() / 255.0,
            grid2D,
            grid3D,
        )
