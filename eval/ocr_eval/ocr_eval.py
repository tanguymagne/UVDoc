import os

import numpy as np
from PIL import Image
from tqdm import tqdm

OCR_FILES_UVDOC = [
    "00001",
    "00002",
    "00003",
    "00009",
    "00010",
    "00017",
    "00018",
    "00019",
    "00021",
    "00023",
    "00029",
    "00032",
    "00033",
    "00034",
    "00036",
    "00037",
    "00041",
    "00045",
    "00046",
    "00048",
]


def OCR_eval_file(filepath, scan_filepath, input_filepath):
    """
    Run OCR on a resized unwarped document image and its corresponding scan.
    """
    import Levenshtein as lv
    import pytesseract
    from jiwer import cer

    # Load original input warped image to get the original image size
    with Image.open(input_filepath) as input_img:
        input_image_shape = input_img.size

    # Performing OCR on both the unwarped document and the scan one
    # Using Neural net LSTM engine only (not legacy). It is a bit better and prevents crashes
    with Image.open(filepath) as our_img:
        OCR_text = pytesseract.image_to_string(our_img.resize(input_image_shape), config="--oem 1")
    with Image.open(scan_filepath) as scan_img:
        GT_text = pytesseract.image_to_string(scan_img, config="--oem 1")

    # Computing metrics
    CER = cer(GT_text, OCR_text)
    ED = lv.distance(OCR_text, GT_text)
    return CER, ED


def OCR_eval_docunet(scan_path, preds_path, input_path):
    """
    Run OCR on the all DocUNet benchmark dataset.
    """
    # Get samples on which to run OCR
    filename = "./eval/ocr_eval/ocr_files.txt"
    with open(filename, "r") as f:
        files = f.readlines()

    # Run OCR on each samples
    CERs, EDs = [], []
    dict_results = {}
    for file in tqdm(files):
        file = file.rstrip("\n").split("/")[-1]
        for i in [1, 2]:
            filepath = os.path.join(preds_path, f"{file}_{i}.png")
            scan_filepath = os.path.join(scan_path, file + ".png")
            input_filepath = os.path.join(input_path, f"{file}_{i} copy.png")

            CER, ED = OCR_eval_file(filepath, scan_filepath, input_filepath)
            CERs.append(CER)
            EDs.append(ED)
            dict_results[f"{file}_{i}"] = dict(CER=CER, ED=ED)

    # Return metrics
    CERmean = np.nanmean(CERs)
    EDmean = np.nanmean(EDs)
    return CERmean, EDmean, dict_results


def OCR_eval_UVDoc(gt_path, preds_path):
    """
    Run OCR on the all UVDoc benchmark dataset.
    """
    # Run OCR on each samples
    CERs, EDs = [], []
    dict_results = {}
    for file in tqdm(OCR_FILES_UVDOC):
        filepath = os.path.join(preds_path, f"{file}.png")
        scan_filepath = os.path.join(gt_path, f"{file}.png")
        input_filepath = os.path.join(preds_path.replace("texture_sample", "warped_textures"), f"{file}.png")

        CER, ED = OCR_eval_file(filepath, scan_filepath, input_filepath)
        CERs.append(CER)
        EDs.append(ED)
        dict_results[file] = dict(CER=CER, ED=ED)

    # Return metrics
    CERmean = np.nanmean(CERs)
    EDmean = np.nanmean(EDs)
    return CERmean, EDmean, dict_results
