import argparse
import os
import glob
from pathlib import Path

import cv2
import numpy as np
import torch

from utils import IMG_SIZE, bilinear_unwarping, load_model


def get_device() -> str:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unwarp_img(
    model: torch.nn.Module, 
    img_path: str, 
    device: str = get_device()
):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    inp = torch.from_numpy(cv2.resize(img, IMG_SIZE).transpose(2, 0, 1)).unsqueeze(0)

    # Make prediction
    inp = inp.to(device)
    point_positions2D, _ = model(inp)

    # Unwarp
    size = img.shape[:2][::-1]
    unwarped = bilinear_unwarping(
        warped_img=torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device),
        point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
        img_size=tuple(size),
    )
    unwarped = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # Save result
    unwarped_BGR = cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR)
    return unwarped_BGR


def infer(ckpt_path:str, img_source:str, img_dest:str):
    """
    Unwarp a document image using the model from ckpt_path.
    """
    device = get_device()

    # Create dir for unwarped images
    os.makedirs(img_dest, exist_ok=True)
    
    # Load model
    model = load_model(ckpt_path)
    model.to(device)
    model.eval()

    if Path(img_source).is_dir():
        img_paths = glob.glob(os.path.join(img_source, '*'))
        for img_path in img_paths:
            unwarped_img = unwarp_img(model, img_path, device)
            cv2.imwrite(
                os.path.join(img_dest, os.path.basename(img_path)), 
                unwarped_img,
            )
    else:
        unwarped_img = unwarp_img(model, img_source, device)
        cv2.imwrite(
            os.path.join(img_dest, os.path.basename(img_source)), 
            unwarped_img
        )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_path", "-c",
        type=str, 
        default="./model/best_model.pkl", 
        help="Path to the model weights as pkl."
    )
    parser.add_argument(
        "--img_source", "-s",
        type=str, 
        help="Path to the document image to unwarp."
    )
    parser.add_argument(
        "--img_dest", "-d", 
        type=str, 
        help="Path to the destinatiom dir for unwarped images."
    )
    
    args = parser.parse_args()
    infer(args.ckpt_path, args.img_source, args.img_dest)
