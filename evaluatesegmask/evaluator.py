import numpy as np
import skimage.io
import skimage.measure
import scipy.optimize
import pandas as pd
import requests
import socket
import os
from importlib.resources import files
from pathlib import Path
import glob
from typing import Union, Dict, List
import re

LEADERBOARD_URL = "https://liveboard-bobiac.onrender.com/update"

def list_ground_truth_images():
    """List all available ground truth mask images.
    
    Returns:
        list: List of available ground truth image filenames
    """
    try:
        # First try package data
        data_path = files('evaluatesegmask').parent / 'data'
        files_list = [f.name for f in data_path.glob('*.png')]
    except Exception:
        # Fallback to local data directory
        data_path = Path(os.path.dirname(os.path.dirname(__file__))) / 'data'
        files_list = [os.path.basename(f) for f in glob.glob(str(data_path / '*.png'))]
    
    if not files_list:
        raise FileNotFoundError("No ground truth images found in data directory")
    
    return sorted(files_list)

def validate_ground_truth_file(filename):
    """Validate that a ground truth file exists.
    
    Args:
        filename (str): Name of the ground truth file to validate
        
    Returns:
        bool: True if file exists, False otherwise
    """
    try:
        path = get_default_gt_path(filename)
        return os.path.exists(path)
    except:
        return False

def get_default_gt_path(ground_truth: str) -> str:
    """Get the path to a ground truth mask file.
    
    Args:
        ground_truth (str): Ground truth number ('001', '002', or '003')
        
    Returns:
        str: Full path to the ground truth file
        
    Raises:
        ValueError: If ground_truth is not one of '001', '002', '003'
    """
    if ground_truth not in ['001', '002', '003']:
        raise ValueError(
            "Ground truth must be one of: '001', '002', '003'\n"
            f"Got: '{ground_truth}'"
        )
        
    try:
        # First try to get it from the installed package data
        data_path = files('evaluatesegmask').parent / 'data' / ground_truth / f"{ground_truth}_masks.png"
        return str(data_path)
    except Exception:
        # Fallback to local data directory if running from source
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', ground_truth, f"{ground_truth}_masks.png")
        if os.path.exists(local_path):
            return local_path
        raise FileNotFoundError(f"Could not find ground truth file {ground_truth}_masks.png in package data or local data directory")

def get_ground_truth_path(ground_truth_folder: str, mask_number: str) -> str:
    """Get the path to a ground truth mask file.
    
    Args:
        ground_truth_folder (str): Ground truth folder ('001', '002', or '003')
        mask_number (str): The mask number (e.g., '008' from '008_masks.png')
        
    Returns:
        str: Full path to the ground truth file
        
    Raises:
        ValueError: If ground_truth_folder is not one of '001', '002', '003'
        FileNotFoundError: If the ground truth file doesn't exist
    """
    if ground_truth_folder not in ['001', '002', '003']:
        raise ValueError(
            "Ground truth folder must be one of: '001', '002', '003'\n"
            f"Got: '{ground_truth_folder}'"
        )
        
    filename = f"{mask_number}_masks.png"
    
    # First try to get it from the installed package data
    try:
        data_path = files('evaluatesegmask').parent / 'data' / ground_truth_folder / filename
        print(f"Looking for ground truth in package data: {data_path}")
        if os.path.exists(str(data_path)):
            print(f"Found ground truth file in package data")
            return str(data_path)
    except Exception as e:
        print(f"Error looking in package data: {e}")
        
    # Fallback to local data directory if running from source
    local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', ground_truth_folder, filename)
    print(f"Looking for ground truth in local data: {local_path}")
    if os.path.exists(local_path):
        print(f"Found ground truth file in local data")
        return local_path
    
    # Try absolute path from current directory
    current_dir_path = os.path.join(os.getcwd(), 'data', ground_truth_folder, filename)
    print(f"Looking for ground truth in current directory: {current_dir_path}")
    if os.path.exists(current_dir_path):
        print(f"Found ground truth file in current directory")
        return current_dir_path
    
    raise FileNotFoundError(
        f"Could not find ground truth file {filename} in ground truth folder {ground_truth_folder}.\n"
        f"Tried paths:\n"
        f"- Package data: {data_path}\n"
        f"- Local data: {local_path}\n"
        f"- Current directory: {current_dir_path}"
    )

def extract_number_from_filename(filename: str) -> str:
    """Extract the number (e.g., '001') from a filename.
    
    Args:
        filename (str): Filename to extract number from
        
    Returns:
        str: Extracted number or None if no match
        
    Example:
        >>> extract_number_from_filename("path/to/001_something.png")
        '001'
    """
    match = re.search(r'(\d{3})', os.path.basename(filename))
    return match.group(1) if match else None

def _intersection_over_union(masks_true, masks_pred):
    """Calculate intersection over union between masks.
    
    Args:
        masks_true (ndarray): ground truth masks
        masks_pred (ndarray): predicted masks
        
    Returns:
        ndarray: IoU matrix of size [n_true+1, n_pred+1]
    """
    n_true = np.max(masks_true) if masks_true.size > 0 else 0
    n_pred = np.max(masks_pred) if masks_pred.size > 0 else 0
    
    iou = np.zeros((n_true + 1, n_pred + 1), dtype=np.float32)
    
    for i in range(1, n_true + 1):
        true_mask = masks_true == i
        for j in range(1, n_pred + 1):
            pred_mask = masks_pred == j
            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            iou[i,j] = intersection / union if union > 0 else 0
            
    return iou

def _true_positive(iou, threshold):
    """Calculate number of true positive matches at a given IoU threshold.
    
    Args:
        iou (ndarray): IoU matrix
        threshold (float): IoU threshold
        
    Returns:
        float: Number of true positive matches
    """
    matches = scipy.optimize.linear_sum_assignment(-iou)
    return np.sum(iou[matches[0], matches[1]] >= threshold)

def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """ 
    Average precision estimation: AP = TP / (TP + FP + FN)

    Args:
        masks_true (list of np.ndarrays (int) or np.ndarray (int)): 
            where 0=NO masks; 1,2... are mask labels
        masks_pred (list of np.ndarrays (int) or np.ndarray (int)): 
            np.ndarray (int) where 0=NO masks; 1,2... are mask labels

    Returns:
        ap (array [len(masks_true) x len(threshold)]): 
            average precision at thresholds
        tp (array [len(masks_true) x len(threshold)]): 
            number of true positives at thresholds
        fp (array [len(masks_true) x len(threshold)]): 
            number of false positives at thresholds
        fn (array [len(masks_true) x len(threshold)]): 
            number of false negatives at thresholds
    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]

    if len(masks_true) != len(masks_pred):
        raise ValueError(
            "metrics.average_precision requires len(masks_true)==len(masks_pred)")

    ap = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))

    for n in range(len(masks_true)):
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k, th in enumerate(threshold):
                tp[n, k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])

    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn

def compute_metrics_for_pair(pred_path: str, gt_path: str, iou_threshold: float = 0.5) -> Dict:
    """Compute metrics for a single prediction-ground truth pair.
    
    Args:
        pred_path (str): Path to prediction mask
        gt_path (str): Path to ground truth mask
        iou_threshold (float): IoU threshold
        
    Returns:
        dict: Metrics for this pair
    """
    pred = skimage.io.imread(pred_path)
    gt = skimage.io.imread(gt_path)
    
    if pred.shape != gt.shape:
        raise ValueError(
            f"Prediction and ground truth dimensions don't match!\n"
            f"Prediction shape: {pred.shape}\n"
            f"Ground truth shape: {gt.shape}\n"
            f"Files: \n- Pred: {pred_path}\n- GT: {gt_path}"
        )

    # Calculate mAP with different IoU thresholds
    ap, tp, fp, fn = average_precision(gt, pred, threshold=[0.5, 0.75, 0.9])
    mean_ap = float(np.mean(ap))

    pred_labels = np.unique(pred[pred > 0])
    gt_labels = np.unique(gt[gt > 0])

    num_pred = len(pred_labels)
    num_gt = len(gt_labels)

    TP_px = int(np.logical_and(pred > 0, gt > 0).sum())
    FP_px = int(np.logical_and(pred > 0, gt == 0).sum())
    FN_px = int(np.logical_and(pred == 0, gt > 0).sum())
    TN_px = int(np.logical_and(pred == 0, gt == 0).sum())

    pixel_accuracy = float((TP_px + TN_px) / (TP_px + FP_px + FN_px + TN_px))
    pixel_f1 = float((2 * TP_px) / (2 * TP_px + FP_px + FN_px)) if (2 * TP_px + FP_px + FN_px) > 0 else 0

    iou_matrix = np.zeros((num_gt, num_pred))
    for i, gt_id in enumerate(gt_labels):
        gt_mask = gt == gt_id
        for j, pred_id in enumerate(pred_labels):
            pred_mask = pred == pred_id
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union > 0:
                iou_matrix[i, j] = intersection / union

    matched_gt, matched_pred = scipy.optimize.linear_sum_assignment(-iou_matrix)
    matches = [(int(i), int(j)) for i, j in zip(matched_gt, matched_pred) if iou_matrix[i, j] >= iou_threshold]

    tp = len(matches)
    fp = num_pred - tp
    fn = num_gt - tp
    f1_instances = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0
    miou = float(np.mean([iou_matrix[i, j] for (i, j) in matches])) if matches else 0

    gt_props = {int(r.label): int(r.area) for r in skimage.measure.regionprops(gt)}
    pred_props = {int(r.label): int(r.area) for r in skimage.measure.regionprops(pred)}
    area_errors = []
    for i, j in matches:
        gt_area = gt_props[int(gt_labels[i])]
        pred_area = pred_props[int(pred_labels[j])]
        abs_error = abs(gt_area - pred_area)
        rel_error = float(abs_error / gt_area) if gt_area > 0 else 0
        area_errors.append((gt_area, pred_area, abs_error, rel_error))

    avg_abs_area_error = float(np.mean([ae[2] for ae in area_errors])) if area_errors else 0
    avg_rel_area_error = float(np.mean([ae[3] for ae in area_errors])) if area_errors else 0

    return {
        "Mean AP": round(mean_ap, 4),
        "Pixel Accuracy": round(pixel_accuracy, 4),
        "Pixel F1 Score": round(pixel_f1, 4),
        "Instance F1 Score": round(f1_instances, 4),
        "Mean IoU": round(miou, 4),
        "Avg. Abs Area Error": round(avg_abs_area_error, 2),
        "Avg. Rel Area Error": round(avg_rel_area_error, 4),
        "TP Instances": int(tp),
        "FP Instances": int(fp),
        "FN Instances": int(fn)
    }

def evaluate_instance_segmentation(pred_path: Union[str, Path], ground_truth: str, iou_threshold: float = 0.5) -> Dict:
    """
    Evaluate instance segmentation predictions against ground truth.
    
    Args:
        pred_path (str): Path to prediction mask image or directory containing predictions
        ground_truth (str): Ground truth folder ('001', '002', or '003') or 'all'
        iou_threshold (float): IoU threshold for considering a match
        
    Returns:
        dict: Dictionary containing all evaluation metrics (averaged if multiple predictions)
        
    Raises:
        ValueError: If ground truth number is invalid or if dimensions don't match
    """
    pred_path = Path(pred_path)
    print(f"\nEvaluating predictions from: {pred_path} (absolute: {pred_path.absolute()})")
    
    # Validate ground truth
    if ground_truth not in ['001', '002', '003', 'all']:
        raise ValueError(
            "Ground truth must be one of: '001', '002', '003', or 'all'\n"
            f"Got: '{ground_truth}'"
        )
    
    # Handle single file case
    if pred_path.is_file():
        if ground_truth == 'all':
            raise ValueError("Cannot use 'all' with a single prediction file. Please specify '001', '002', or '003'.")
        
        # Get the number from the prediction filename
        pred_number = extract_number_from_filename(str(pred_path))
        if not pred_number:
            raise ValueError(f"Could not extract number from prediction filename: {pred_path}")
            
        try:
            gt_path = get_ground_truth_path(ground_truth, pred_number)
            results = compute_metrics_for_pair(str(pred_path), gt_path, iou_threshold)
            
            metrics_table = pd.DataFrame({
                "Metric": list(results.keys()),
                "Value": list(results.values())
            })
            print(metrics_table.to_string(index=False))
            return results
        except FileNotFoundError as e:
            raise ValueError(f"No matching ground truth found for prediction {pred_path.name} in folder {ground_truth}")
    
    # Handle directory case
    if not pred_path.is_dir():
        raise ValueError(f"Path does not exist or is neither a file nor directory: {pred_path}")
    
    # Get all PNG files in the prediction directory
    pred_files = list(pred_path.glob("**/*.png"))
    print(f"\nFound prediction files: {[f.name for f in pred_files]}")
    
    if not pred_files:
        raise ValueError(f"No PNG files found in directory: {pred_path}")
    
    # Process each prediction file
    all_results = []
    ground_truth_nums = ['001', '002', '003'] if ground_truth == 'all' else [ground_truth]
    
    for gt_folder in ground_truth_nums:
        print(f"\nProcessing ground truth folder: {gt_folder}")
        # For each prediction, try to find its corresponding ground truth
        for pred_file in pred_files:
            print(f"\nProcessing prediction file: {pred_file.name}")
            pred_number = extract_number_from_filename(str(pred_file))
            if not pred_number:
                print(f"Warning: Could not extract number from prediction filename: {pred_file.name}")
                continue
                
            try:
                gt_path = get_ground_truth_path(gt_folder, pred_number)
                results = compute_metrics_for_pair(str(pred_file), gt_path, iou_threshold)
                all_results.append(results)
                print(f"\nResults for {pred_file.name} vs {os.path.basename(gt_path)}:")
                metrics_table = pd.DataFrame({
                    "Metric": list(results.keys()),
                    "Value": list(results.values())
                })
                print(metrics_table.to_string(index=False))
            except FileNotFoundError as e:
                print(f"Warning: No matching ground truth found for {pred_file.name} in folder {gt_folder}")
            except Exception as e:
                print(f"Error processing {pred_file.name}: {str(e)}")
    
    if not all_results:
        raise ValueError("No valid prediction-ground truth pairs were found to evaluate")
    
    # Compute averages
    avg_results = {}
    for metric in all_results[0].keys():
        values = [r[metric] for r in all_results]
        avg_results[metric] = round(float(np.mean(values)), 4)
    
    print("\nAVERAGE RESULTS:")
    metrics_table = pd.DataFrame({
        "Metric": list(avg_results.keys()),
        "Value": list(avg_results.values())
    })
    print(metrics_table.to_string(index=False))
    
    return avg_results

def post_to_leaderboard(name, results_dict):
    """
    Post results to the leaderboard.
    
    Args:
        name (str): Name or team ID for the leaderboard
        results_dict (dict): Dictionary containing evaluation metrics
    """
    payload = {
        "name": name,
        "host": socket.gethostname(),
        "results": results_dict
    }
    try:
        res = requests.post(LEADERBOARD_URL, json=payload, timeout=5)
        if res.ok:
            print("✅ Submitted to leaderboard.")
        else:
            print(f"❌ Submission failed: {res.status_code}")
    except Exception as e:
        print(f"⚠️ Failed to connect to leaderboard: {e}") 

def evaluate(pred_path: Union[str, Path], name: str, ground_truth: str, iou_threshold: float = 0.5) -> Dict:
    """
    Evaluate instance segmentation and post results to leaderboard in one step.
    
    Args:
        pred_path (str): Path to prediction mask image or directory containing predictions
        name (str): Name or team ID for the leaderboard
        ground_truth (str): Ground truth number ('001', '002', or '003') or 'all'
        iou_threshold (float): IoU threshold for considering a match
        
    Returns:
        dict: Dictionary containing all evaluation metrics (averaged if multiple predictions)
        
    Raises:
        ValueError: If ground truth number is invalid or if dimensions don't match
    """
    results = evaluate_instance_segmentation(pred_path, ground_truth, iou_threshold)
    post_to_leaderboard(name, results)
    return results 