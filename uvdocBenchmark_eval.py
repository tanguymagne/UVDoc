import argparse
import json
import multiprocessing as mp
import os

from utils import get_version

N_LINES = 25


def visual_metrics_process(queue, uvdoc_path, preds_path, verbose):
    """
    Subprocess function that computes visual metrics (MS-SSIM, LD, and AD) based on a matlab script.
    """
    import matlab.engine

    eng = matlab.engine.start_matlab()
    eng.cd(r"./eval/eval_code/", nargout=0)

    mean_ms, mean_ad = eng.evalScriptUVDoc(uvdoc_path, preds_path, verbose, nargout=2)
    queue.put(dict(ms=mean_ms, ad=mean_ad))


def ocr_process(queue, uvdoc_path, preds_path):
    """
    Subprocess function that computes OCR metrics (CER and ED).
    """
    from eval.ocr_eval.ocr_eval import OCR_eval_UVDoc

    CERmean, EDmean, OCR_dict_results = OCR_eval_UVDoc(uvdoc_path, preds_path)
    with open(os.path.join(preds_path, "ocr_res.json"), "w") as f:
        json.dump(OCR_dict_results, f)
    queue.put(dict(cer=CERmean, ed=EDmean))


def new_line_metric_process(queue, uvdoc_path, preds_path, n_lines):
    """
    Subprocess function that computes the new line metrics on the UVDoc benchmark.
    """
    from uvdocBenchmark_metric import compute_line_metric

    hor_metric, ver_metric = compute_line_metric(uvdoc_path, preds_path, n_lines)
    queue.put(dict(hor_line=hor_metric, ver_line=ver_metric))


def compute_metrics(uvdoc_path, pred_path, pred_type, verbose=False):
    """
    Compute and save all metrics.
    """
    if not pred_path.endswith("/"):
        pred_path += "/"
    q = mp.Queue()

    # Create process to compute MS-SSIM, LD, AD
    p1 = mp.Process(
        target=visual_metrics_process,
        args=(q, os.path.join(uvdoc_path, "texture_sample"), os.path.join(pred_path, pred_type), verbose),
    )
    p1.start()

    # Create process to compute new line metrics
    p2 = mp.Process(
        target=new_line_metric_process,
        args=(q, uvdoc_path, os.path.join(pred_path, "bm"), N_LINES),
    )
    p2.start()

    # Create process to compute OCR metrics
    p3 = mp.Process(
        target=ocr_process, args=(q, os.path.join(uvdoc_path, "texture_sample"), os.path.join(pred_path, pred_type))
    )
    p3.start()

    p1.join()
    p2.join()
    p3.join()

    # Get results
    res = {}
    for _ in range(q.qsize()):
        ret = q.get()
        for k, v in ret.items():
            res[k] = v

    # Print and saves results
    print("--- Results ---")
    print(f"  Mean MS-SSIM      : {res['ms']}")
    print(f"  Mean AD           : {res['ad']}")
    print(f"  Mean CER          : {res['cer']}")
    print(f"  Mean ED           : {res['ed']}")
    print(f"  Hor Line          : {res['hor_line']}")
    print(f"  Ver Line          : {res['ver_line']}")

    with open(os.path.join(pred_path, pred_type, "resUVDoc.txt"), "w") as f:
        f.write(f"Mean MS-SSIM      : {res['ms']}\n")
        f.write(f"Mean AD           : {res['ad']}\n")
        f.write(f"Mean CER          : {res['cer']}\n")
        f.write(f"Mean ED           : {res['ed']}\n")
        f.write(f"Hor Line          : {res['hor_line']}\n")
        f.write(f"Ver Line          : {res['ver_line']}\n")

        f.write("\n--- Module Version ---\n")
        for module, version in get_version().items():
            f.write(f"{module:25s}: {version}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--uvdoc-path", type=str, default="./data/UVDoc_benchmark/", help="Path to the uvdoc benchmark dataset"
    )
    parser.add_argument("--pred-path", type=str, help="Path to the UVDoc benchmark predictions. Need to be absolute.")
    parser.add_argument(
        "--pred-type",
        type=str,
        default="uwp_texture",
        choices=["uwp_texture", "uwp_img"],
        help="Which type of prediction to compare. Either the unwarped textures or the unwarped litted images.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    compute_metrics(
        uvdoc_path=os.path.abspath(args.uvdoc_path),
        pred_path=os.path.abspath(args.pred_path),
        pred_type=args.pred_type,
        verbose=args.verbose,
    )
