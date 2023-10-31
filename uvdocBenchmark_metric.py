import json
import os
from os.path import join as pjoin

import hdf5storage as h5
import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import binary_erosion
from tqdm import tqdm

from utils import bilinear_unwarping_from_numpy

WIDTH = 1000
HEIGHT = 1000


def create_vertical_stripe_texture(width, height, stripe_width=1, position=0):
    """
    Create an image with a vertical stripe.
    """
    im = np.ones((height, width, 3), dtype=np.uint8) * 255
    im[:, position : position + stripe_width] = 0
    return im


def create_horizontal_stripe_texture(width, height, stripe_width=1, position=0):
    """
    Create an image with a horizontal stripe.
    """
    im = np.ones((height, width, 3), dtype=np.uint8) * 255
    im[position : position + stripe_width, :] = 0
    return im


def warp_texture(texture, uvmap):
    """
    Warp an input texture based on the provided uvmap.
    """
    # Warp the texture based on the uv
    torch_texture_unwarp = torch.from_numpy(np.expand_dims(texture.transpose(2, 0, 1), axis=0)).float()
    uvmap_torch = torch.from_numpy(np.expand_dims(uvmap * 2 - 1, axis=0)).float()
    warped_texture = F.grid_sample(torch_texture_unwarp, uvmap_torch, align_corners=False)
    warped_texture = np.clip(warped_texture[0].numpy().transpose(1, 2, 0), 0, 255) / 255

    # Postprocessing to have nicer results
    grey = np.all(warped_texture == 0.5, axis=-1)
    warped_texture[grey] = np.nan
    mask = 1 - np.all(np.isnan(warped_texture), axis=-1).astype(int)
    mask_small = binary_erosion(mask).astype(int)
    mask_small = np.expand_dims(mask_small, axis=-1)
    warped_texture[np.repeat(~mask_small.astype(bool), 3, axis=-1)] = 1
    warped_texture = (warped_texture * 255).astype(np.uint8)

    return warped_texture


def compute_metric_single_line(uvmap, bm, pos, direction="horizontal"):
    """
    Compute the line metric for a single line.
    args:
        uvmap: uvmap of the document, shape (height, width, 2)
        bm: predicted backward mapping, shape (height, width, 2)
        pos: position of the line to compute the metric
        direction: direction of the line to compute the metric (horizontal or vertical)
    """
    # Create the original straight line
    if direction == "horizontal":
        stripe = create_horizontal_stripe_texture(WIDTH, HEIGHT, stripe_width=1, position=pos)
    elif direction == "vertical":
        stripe = create_vertical_stripe_texture(WIDTH, HEIGHT, stripe_width=1, position=pos)
    else:
        raise ValueError("Direction must be horizontal or vertical")

    # Warp the stripe according to the ground truth uvmap and unwarp it according to the predicted bm
    warped_stripe = warp_texture(stripe, uvmap)
    unwarped_stripe = bilinear_unwarping_from_numpy(warped_stripe.astype(float) / 255.0, bm, (WIDTH, HEIGHT))

    # Binarize the result
    THRESH = 0.5
    unwarped_stripe = unwarped_stripe[:, :, 0]
    unwarped_stripe[unwarped_stripe < THRESH] = 0
    unwarped_stripe[unwarped_stripe >= THRESH] = 1

    # Find the black pixels
    xs, ys = np.where(unwarped_stripe == 0)
    if len(xs) == 0 or len(ys) == 0:
        # No black pixels in the line, this means that the backward mapping is pretty bad
        return np.nan

    # Compute the metric
    if direction == "horizontal":
        return np.std(xs)
    elif direction == "vertical":
        return np.std(ys)


def compute_sample_line_metric(uvdoc_path, pred_path, sample, n_lines):
    """
    Compute all lines metric for a given sample.
    """
    # Load ground truth UVmap
    metadata_path = pjoin(uvdoc_path, "metadata_sample", f"{sample}.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    uvmap_path = pjoin(uvdoc_path, "uvmap", f"{metadata['geom_name']}.mat")
    uvmap = h5.loadmat(uvmap_path)["uv"]

    # Load predicted backward mapping
    bm_path = pjoin(pred_path, f"{sample}.mat")
    bm = h5.loadmat(bm_path)["bm"]

    # Compute metric
    stds_hor = []
    stds_ver = []
    for pos in np.linspace(50, 950, n_lines, dtype=int):
        uvmap = h5.loadmat(uvmap_path)["uv"]
        stds_hor.append(compute_metric_single_line(uvmap, bm, pos, direction="horizontal"))
        stds_ver.append(compute_metric_single_line(uvmap, bm, pos, direction="vertical"))

    return np.nanmean(stds_hor), np.nanmean(stds_ver)


def compute_line_metric(uvdoc_path, pred_path, n_lines=25):
    """
    Compute the line metric over the whole UVDoc dataset.
    """
    # Find all samples
    all_samples = sorted([x[:-4] for x in os.listdir(pjoin(uvdoc_path, "img"))])

    # Compute the metric for each sample
    lines = []
    cols = []
    for sample in tqdm(all_samples):
        hor, ver = compute_sample_line_metric(uvdoc_path, pred_path, sample, n_lines)
        lines.append(hor)
        cols.append(ver)

    # Saves all results including individual ones
    with open(os.path.join(pred_path, "line_metric.json"), "w") as f:
        json.dump(
            {sample: {"hor": lines[i], "ver": cols[i]} for i, sample in enumerate(all_samples)},
            f,
        )

    with open(os.path.join(pred_path, "line_metric_mean.json"), "w") as f:
        json.dump(
            {"hor": np.mean(lines), "ver": np.mean(cols)},
            f,
        )

    return np.mean(lines), np.mean(cols)
