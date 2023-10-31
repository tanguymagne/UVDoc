import argparse
import os

import cv2
import numpy as np
import torch

from utils import IMG_SIZE, bilinear_unwarping, load_model


def unwarp_img(ckpt_path, img_path, img_size):
    """
    Unwarp a document image using the model from ckpt_path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(ckpt_path)
    model.to(device)
    model.eval()

    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    inp = torch.from_numpy(cv2.resize(img, img_size).transpose(2, 0, 1)).unsqueeze(0)

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
    cv2.imwrite(os.path.splitext(img_path)[0] + "_unwarp.png", unwarped_BGR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt-path", type=str, default="./model/best_model.pkl", help="Path to the model weights as pkl."
    )
    parser.add_argument("--img-path", type=str, help="Path to the document image to unwarp.")

    args = parser.parse_args()

    unwarp_img(args.ckpt_path, args.img_path, IMG_SIZE)
