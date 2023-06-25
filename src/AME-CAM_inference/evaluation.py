import numpy as np
from medpy.metric import dc, hd95

def compute_dice(gt, pred):
    num_gt = np.sum(gt)
    num_pred = np.sum(pred)
    if num_gt == 0:
        if num_pred == 0:
            return 1
        else:
            return 0
    else:
        return dc(pred, gt)

def compute_mIOU(gt, pred):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou_score = (np.sum(intersection) + 1e-5) / (np.sum(union) + 1e-5)
    return iou_score

def compute_HD95(gt, pred):
    num_gt = np.sum(gt)
    num_pred = np.sum(pred)
    if num_gt == 0 and num_pred == 0:
        return 0
    if num_gt == 0 or num_pred == 0:
        return 373.12866
    return hd95(pred, gt, (1, 1))

def compute_seg_metrics(gt, pred):
    result = {}
    gt = gt.astype(np.uint8)
    pred = pred.astype(np.uint8)
    result['Dice'] = compute_dice(gt, pred)
    result['IoU'] = compute_mIOU(gt, pred)
    result['HD95'] = compute_HD95(gt, pred)
    return result