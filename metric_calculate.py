import os
import numpy as np

def metrics_iou(pr, gt):
    '''
    calculate the iou for the individual category
    :param pr: the predicted mask or logit with shape of (h, w)
    :param gt: the ground truth of mask with shape of (h, w)
    :return:
    '''
    pr = np.int8(pr > 0.5)
    gt = np.int8(gt > 0)
    intersection = np.sum(gt * pr)
    union = np.sum(gt) + np.sum(pr) - intersection
    return intersection / union

def mean_iou(pr, gt):
    '''
    calcalate the mean IOU
    :param pr: the predicted mask or logit with shape of (c, h, w), where c is the category number
    :param gt: the ground truth of mask with shape of (c, h, w)
    :return:
    '''
    c = pr.shape[0]
    ious = []
    for i in range(c):
        iou = metrics_iou(pr[i], gt[i])
        ious.append(iou)
    return np.mean(ious)

if __name__ == '__main__':
    # mask path
    path_mask = 'validation/mask'
    # prediction path
    path_prediction = 'validation/prediction'

    mean_metric = 0
    count = 0
    for root, dirs, files in os.walk(path_mask):
        for file in files:
            if file.endswith('.npy'):
                temp_mask = np.load(os.path.join(path_mask, file))
                temp_prediction = np.load(os.path.join(path_prediction, file))
                mean_metric += mean_iou(temp_prediction, temp_mask)
                count += 1
    mean_metric = mean_metric / count
    print(mean_metric)