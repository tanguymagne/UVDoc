import argparse
import os

import cv2
import hdf5storage as h5
import numpy as np
import torch
from tqdm import tqdm

from utils import IMG_SIZE, bilinear_unwarping, load_model


class UVDocBenchmarkLoader(torch.utils.data.Dataset):
    """
    Torch dataset class for the UVDoc benchmark dataset.
    """

    def __init__(
        self,
        data_path,
        img_size=(488, 712),
    ):
        self.dataroot = data_path
        self.im_list = os.listdir(os.path.join(self.dataroot, "img"))
        self.img_size = img_size

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        im_name = self.im_list[index]
        img_path = os.path.join(self.dataroot, "img", im_name)
        img_RGB = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_RGB = torch.from_numpy(cv2.resize(img_RGB, self.img_size).transpose(2, 0, 1))
        return img_RGB, im_name


def infer_uvdoc(model, dataloader, device, save_path):
    """
    Unwarp all images in the UVDoc benchmark and save them, along with the mappings.
    """
    model.eval()

    os.makedirs(os.path.join(save_path, "uwp_img"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "bm"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "uwp_texture"), exist_ok=True)

    for img_RGB, im_names in tqdm(dataloader):
        # Inference
        img_RGB = img_RGB.to(device)
        point_positions2D, _ = model(img_RGB)

        # Warped image need to be re-open to get full resolution (downsampled in data loader)
        warped = cv2.imread(os.path.join(dataloader.dataset.dataroot, "img", im_names[0]))
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        warped = torch.from_numpy(warped.transpose(2, 0, 1) / 255.0).float()
        size = warped.shape[1:][::-1]

        # Unwarping
        unwarped = bilinear_unwarping(
            warped_img=torch.unsqueeze(warped, dim=0).to(device),
            point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
            img_size=tuple(size),
        )
        unwarped = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        unwarped_BGR = cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR)

        cv2.imwrite(
            os.path.join(save_path, "uwp_img", im_names[0].split(" ")[0].split(".")[0] + ".png"),
            unwarped_BGR,
        )

        # Unwarp and save the texture
        warp_texture = cv2.imread(os.path.join(dataloader.dataset.dataroot, "warped_textures", im_names[0]))
        warp_texture = cv2.cvtColor(warp_texture, cv2.COLOR_BGR2RGB)
        warp_texture = torch.from_numpy(warp_texture.transpose(2, 0, 1) / 255.0).float()
        size = warp_texture.shape[1:][::-1]

        unwarped_texture = bilinear_unwarping(
            warped_img=torch.unsqueeze(warp_texture, dim=0).to(device),
            point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
            img_size=tuple(size),
        )
        unwarped_texture = (unwarped_texture[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        unwarped_texture_BGR = cv2.cvtColor(unwarped_texture, cv2.COLOR_RGB2BGR)

        cv2.imwrite(
            os.path.join(save_path, "uwp_texture", im_names[0].split(" ")[0].split(".")[0] + ".png"),
            unwarped_texture_BGR,
        )

        # Save Backward Map
        h5.savemat(
            os.path.join(save_path, "bm", im_names[0].split(" ")[0].split(".")[0] + ".mat"),
            {"bm": point_positions2D[0].detach().cpu().numpy().transpose(1, 2, 0)},
        )


def create_uvdoc_results(ckpt_path, uvdoc_path, img_size):
    """
    Create results for the UVDoc benchmark.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model, create dataset and save directory
    model = load_model(ckpt_path)
    model.to(device)

    dataset = UVDocBenchmarkLoader(data_path=uvdoc_path, img_size=img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    save_path = os.path.join("/".join(ckpt_path.split("/")[:-1]), "output_uvdoc")
    os.makedirs(save_path, exist_ok=True)
    print(f"    Results will be saved at {save_path}", flush=True)

    # Infer results
    infer_uvdoc(model, dataloader, "cuda:0", save_path)
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-path", type=str, default="./model/best_model.pkl", help="Path to the model weights as pkl."
    )
    parser.add_argument(
        "--uvdoc-path", type=str, default="./data/UVDoc_benchmark/", help="Path to the UVDocBenchmark dataset."
    )
    args = parser.parse_args()

    create_uvdoc_results(args.ckpt_path, os.path.abspath(args.uvdoc_path), IMG_SIZE)
