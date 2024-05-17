import os
import glob
import numpy as np
from PIL import Image

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def compute_miou(label_trues, label_preds, n_class=20):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)   # n_class, n_class
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-5)
    print(iu, 'avg:', np.mean(iu))
    mean_iu_noBG = np.nanmean(iu[1:])
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, mean_iu_noBG, fwavacc


Pred_glob = os.path.join('/media/meng2/disk11/ZRT/PycharmProjects/AttrCLIP/visualization/check',"*.png")
Pred_name_list = []
Pred_name_list.extend(glob.glob(Pred_glob))

Gt_name_list = []
content = open('/media/meng2/disk11/ZRT/dataset/VSAC/base/val.txt','r').readlines()
for item in content:
    Gt_name_list.append(item.strip('\n'))

pred_list=[]
gt_list=[]
for idx, name in enumerate(Gt_name_list):
    gt_path = '/media/meng2/disk11/ZRT/dataset/VSAC/base/attribute/%s.png'%name
    gt = np.array(Image.open(gt_path))
    gt_list.append(gt)

    pred_path = '/media/meng2/disk11/ZRT/PycharmProjects/AttrCLIP/visualization/check_vsac_attr20_bg/%s.png'%idx
    pred = np.array(Image.open(pred_path))
    pred_list.append(pred)

acc, acc_cls, mean_iu, mean_iu_noBG, fwavacc = compute_miou(gt_list, pred_list)
print(acc, acc_cls, mean_iu, mean_iu_noBG, fwavacc)