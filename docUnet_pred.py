import argparse
import os
import platform
import re
import subprocess
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils import IMG_SIZE, bilinear_unwarping, load_model


def get_processor_name():
    """
    Returns information about the processor used.
    Taken from https://stackoverflow.com/a/13078519.
    """
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)
    return ""


def count_parameters(model):
    """
    Returns the number of parameters of a model.
    Taken from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class docUnetLoader(torch.utils.data.Dataset):
    """
    Torch dataset class for the DocUNet benchmark dataset.
    """

    def __init__(
        self,
        data_path,
        crop="original",
        img_size=(488, 712),
    ):
        self.dataroot = data_path
        self.crop = crop
        self.im_list = os.listdir(os.path.join(self.dataroot, self.crop))
        self.img_size = img_size

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        im_name = self.im_list[index]
        img_path = os.path.join(self.dataroot, self.crop, im_name)
        img_RGB = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_RGB = torch.from_numpy(cv2.resize(img_RGB, tuple(self.img_size)).transpose(2, 0, 1))
        return img_RGB, im_name


def infer_docUnet(model, dataloader, device, save_path):
    """
    Unwarp all images in the DocUNet benchmark and save them.
    Also measure the times it takes to perform this operation.
    """
    model.eval()
    inference_times = []
    inferenceGPU_times = []
    for img_RGB, im_names in tqdm(dataloader):
        # Inference
        start_toGPU = time.time()
        img_RGB = img_RGB.to(device)
        start_inf = time.time()
        point_positions2D, _ = model(img_RGB)
        end_inf = time.time()

        # Warped image need to be re-open to get full resolution (downsampled in data loader)
        warped = cv2.imread(os.path.join(dataloader.dataset.dataroot, dataloader.dataset.crop, im_names[0]))
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        warped = torch.from_numpy(warped.transpose(2, 0, 1) / 255.0).float()

        # To unwarp using the GT aspect ratio, uncomment following lines and replace
        # `size = warped.shape[:2]` by `size = gt.shape[:2]`
        # gt = cv2.imread(
        #     os.path.join(
        #         dataloader.dataset.dataroot,
        #         "scan",
        #         im_names[0].split("_")[0] + ".png",
        #     )
        # )
        size = warped.shape[1:][::-1]

        # Unwarping
        start_unwarp = time.time()
        unwarped = bilinear_unwarping(
            warped_img=torch.unsqueeze(warped, dim=0).to(device),
            point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
            img_size=tuple(size),
        )

        end_unwarp = time.time()
        unwarped = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        unwarped_BGR = cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR)
        end_toGPU = time.time()

        cv2.imwrite(
            os.path.join(save_path, im_names[0].split(" ")[0].split(".")[0] + ".png"),
            unwarped_BGR,
        )

        inference_times.append(end_inf - start_inf + end_unwarp - start_unwarp)
        inferenceGPU_times.append(end_inf - start_toGPU + end_toGPU - start_unwarp)

    # Computes average inference time and the number of parameters of the model
    avg_inference_time = np.mean(inference_times)
    avg_inferenceGPU_time = np.mean(inferenceGPU_times)
    n_params = count_parameters(model)
    return avg_inference_time, avg_inferenceGPU_time, n_params


def create_results(ckpt_path, docUnet_path, crop, img_size):
    """
    Create results for the DocUNet benchmark.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model, create dataset and save directory
    model = load_model(ckpt_path)
    model.to(device)

    dataset = docUnetLoader(docUnet_path, crop, img_size=img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    save_path = os.path.join("/".join(ckpt_path.split("/")[:-1]), "docunet", crop)
    os.makedirs(save_path, exist_ok=False)
    print(f"    Results will be saved at {save_path}", flush=True)

    # Infer results from the model and saves metadata
    inference_time, inferenceGPU_time, n_params = infer_docUnet(model, dataloader, device, save_path)
    with open(os.path.join(save_path, "model_info.txt"), "w") as f:
        f.write("\n---Model and Hardware Information---\n")
        f.write(f"Inference Time : {inference_time:.5f}s\n")
        f.write(f"  FPS : {1/inference_time:.1f}\n")
        f.write(f"Inference Time (Include Loading To/From GPU) : {inferenceGPU_time:.5f}s\n")
        f.write(f"  FPS : {1/inferenceGPU_time:.1f}\n")
        f.write("Using :\n")
        f.write(f"  CPU : {get_processor_name()}\n")
        f.write(f"  GPU : {torch.cuda.get_device_name(0)}\n")
        f.write(f"Number of Parameters : {n_params:,}\n")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt-path", type=str, default="./model/best_model.pkl", help="Path to the model weights as pkl."
    )
    parser.add_argument("--docunet-path", type=str, default="./data/DocUNet", help="Path to the docunet benchmark.")
    parser.add_argument(
        "--crop-type",
        type=str,
        default="crop",
        help="The type of cropping to use as input of the model : 'crop' or 'original'.",
    )

    args = parser.parse_args()

    create_results(args.ckpt_path, os.path.abspath(args.docunet_path), args.crop_type, IMG_SIZE)
