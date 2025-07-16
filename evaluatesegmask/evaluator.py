import numpy as np
import skimage.io
import skimage.measure
import scipy.optimize
import pandas as pd
import requests
import socket

LEADERBOARD_URL = "https://liveboard-bobiac.onrender.com/update"

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

    # Calculate mAP with different IoU thresholds
    ap, tp, fp, fn = average_precision(gt, pred, threshold=[0.5, 0.75, 0.9])
    mean_ap = float(np.mean(ap))  # Convert to native Python float

    pred_labels = np.unique(pred[pred > 0])
    gt_labels = np.unique(gt[gt > 0])

    num_pred = len(pred_labels)
    num_gt = len(gt_labels)

    TP_px = int(np.logical_and(pred > 0, gt > 0).sum())  # Convert to native Python int
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

    results = {
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

def evaluate(pred_path, name, gt_path="../_static/images/student_group/001_masks.png", iou_threshold=0.5):
    """
    Evaluate instance segmentation and post results to leaderboard in one step.
    
    Args:
        pred_path (str): Path to the prediction mask image
        name (str): Name or team ID for the leaderboard
        gt_path (str): Path to the ground truth mask image
        iou_threshold (float): IoU threshold for considering a match
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    results = evaluate_instance_segmentation(pred_path, gt_path, iou_threshold)
    post_to_leaderboard(name, results)
    return results 