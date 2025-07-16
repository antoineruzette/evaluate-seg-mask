import numpy as np
import skimage.io
import skimage.measure
import scipy.optimize
import pandas as pd
import requests
import socket

LEADERBOARD_URL = "https://liveboard-bobiac.onrender.com/update"

def evaluate_instance_segmentation(pred_path, gt_path="../_static/images/student_group/001_masks.png", iou_threshold=0.5):
    """
    Evaluate instance segmentation predictions against ground truth.
    
    Args:
        pred_path (str): Path to the prediction mask image
        gt_path (str): Path to the ground truth mask image
        iou_threshold (float): IoU threshold for considering a match
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    pred = skimage.io.imread(pred_path)
    gt = skimage.io.imread(gt_path)

    pred_labels = np.unique(pred[pred > 0])
    gt_labels = np.unique(gt[gt > 0])

    num_pred = len(pred_labels)
    num_gt = len(gt_labels)

    TP_px = np.logical_and(pred > 0, gt > 0).sum()
    FP_px = np.logical_and(pred > 0, gt == 0).sum()
    FN_px = np.logical_and(pred == 0, gt > 0).sum()
    TN_px = np.logical_and(pred == 0, gt == 0).sum()

    pixel_accuracy = (TP_px + TN_px) / (TP_px + FP_px + FN_px + TN_px)
    pixel_f1 = (2 * TP_px) / (2 * TP_px + FP_px + FN_px) if (2 * TP_px + FP_px + FN_px) > 0 else 0

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
    matches = [(i, j) for i, j in zip(matched_gt, matched_pred) if iou_matrix[i, j] >= iou_threshold]

    tp = len(matches)
    fp = num_pred - tp
    fn = num_gt - tp
    f1_instances = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    miou = np.mean([iou_matrix[i, j] for (i, j) in matches]) if matches else 0

    gt_props = {r.label: r.area for r in skimage.measure.regionprops(gt)}
    pred_props = {r.label: r.area for r in skimage.measure.regionprops(pred)}
    area_errors = []
    for i, j in matches:
        gt_area = gt_props[gt_labels[i]]
        pred_area = pred_props[pred_labels[j]]
        abs_error = abs(gt_area - pred_area)
        rel_error = abs_error / gt_area if gt_area > 0 else 0
        area_errors.append((gt_area, pred_area, abs_error, rel_error))

    avg_abs_area_error = np.mean([ae[2] for ae in area_errors]) if area_errors else 0
    avg_rel_area_error = np.mean([ae[3] for ae in area_errors]) if area_errors else 0

    results = {
        "Pixel Accuracy": round(pixel_accuracy, 4),
        "Pixel F1 Score": round(pixel_f1, 4),
        "Instance F1 Score": round(f1_instances, 4),
        "Mean IoU": round(miou, 4),
        "Avg. Abs Area Error": round(avg_abs_area_error, 2),
        "Avg. Rel Area Error": round(avg_rel_area_error, 4),
        "TP Instances": tp,
        "FP Instances": fp,
        "FN Instances": fn
    }

    metrics_table = pd.DataFrame({
        "Metric": list(results.keys()),
        "Value": list(results.values())
    })
    
    print(metrics_table.to_string(index=False))
    return results

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