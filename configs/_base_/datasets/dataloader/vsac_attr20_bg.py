import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp
import warnings
from collections import OrderedDict
import clip
import torch
import mmcv
import numpy as np
from PIL import Image
from preprocess import tools
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import Compose, LoadAnnotations
from PIL import Image

def getPartPalette(mode):
    '''
    http://blog.csdn.net/yhl_leo/article/details/52185581
    '''
    if mode == 'color':
        pal = np.array([[0, 0, 0],
                        [128, 0, 0],
                        [0, 128, 0],
                        [128, 128, 0],
                        [0, 0, 128],
                        [128, 0, 128],
                        [0, 128, 128],
                        [128, 128, 128],
                        [64, 0, 0],
                        [192, 0, 0],
                        [64, 128, 0],
                        [192, 128, 0],
                        [64, 0, 128],
                        [192, 0, 128],
                        [64, 128, 128],
                        [192, 128, 128],
                        [0, 64, 0],
                        [128, 64, 0],
                        [0, 192, 0],
                        [128, 192, 0],
                        [0, 64, 128],
                       [128, 32, 32],
                       [32, 128, 32],
                       [128, 128, 32],
                       [32, 32, 128],
                       [128, 32, 128],
                       [32, 128, 128],
                       [128, 128, 128],
                       [64, 32, 32],
                       [192, 32, 32],
                       [64, 128, 32],
                       [192, 128, 32],
                       [64, 32, 128],
                       [192, 32, 128],
                       [64, 128, 128],
                       [192, 128, 128],
                       [32, 64, 32],
                       [128, 64, 32],
                       [32, 192, 32],
                       [128, 192, 32],
                       [32, 64, 128],
                       [128, 16, 16],
                       [16, 128, 16],
                       [128, 128, 16],
                       [16, 16, 128],
                       [128, 16, 128],
                       [16, 128, 128],
                       [128, 128, 128],
                       [64, 16, 16],
                       [192, 16, 16]], dtype='uint8').flatten()
    elif mode =='binary':
        pal = np.array([[0, 0, 0],
                        [255, 255, 255]], dtype='uint8').flatten()
    return pal

def colorize_mask(mask, mode):
    """
    :param mask: 图片大小的数值，代表不同的颜色
    :return:
    """
    new_mask = Image.fromarray(mask.astype(np.uint8), 'P')  # 将二维数组转为图像

    pal = getPartPalette(mode)
    new_mask.putpalette(pal)
    # print(new_mask.show())
    return new_mask

def labelTopng(label, img_name, mode='color'):
    '''
    convert tensor cpu label to png and save
    '''
    #label = label.numpy()             # 320 320
    label_pil = colorize_mask(label, mode)
    label_pil.save(img_name)

@DATASETS.register_module()
class VSACDatasetAttr20BG(CustomDataset):
    """Pascal VOC dataset.
    Args:
        split (str): Split txt file for Pascal VOC and exclude "background" class.
    """

    CLASSES = ('background', 'torso', 'head', 'tail', 'leg', 'horn',
               'muzzle', 'window', 'door', 'mirror', 'cabin',
               'headlight', 'wheel', 'locomotive', 'handle bar', 'saddle',
               'aircraft wing', 'aircraft tail', 'fuselage', 'frame')

    PALETTE = [[0,0,0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
               [128, 64, 0],[0, 192, 0],[128, 192, 0]]

    def __init__(self, split, **kwargs):
        super(VSACDatasetAttr20BG, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=True,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

    def evaluate_classes(self,
                 seen_idx,
                 unseen_idx,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                    results or predict segmentation map for computing evaluation
                    metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        Gt_name_list = []
        content = open('/data1/zhang_runtong/datasets/VSAC/novel/val.txt', 'r').readlines()
        for item in content:
            Gt_name_list.append(item.strip('\n'))

        pred_list = []
        gt_list = []

        device = "cuda" if torch.cuda.is_available() else "cpu"
        CLIP, preprocess = clip.load("ViT-B/16", device=device)
        text = clip.tokenize(["a cat", "a dog", "a horse", "a cow", "a sheep",
                              "an aeroplane", "a bicycle", "a motorcycle", "a car",
                              "a bus", "a train", "a bear", "an elephant", "a giraffe",
                              "a zebra", "a truck"]).to(device)

        for idx, pred in enumerate(results):
            name = Gt_name_list[idx]
            pred = pred.astype(np.uint8)
            img = np.array(Image.open('/data1/zhang_runtong/datasets/VSAC/novel/JPEGImages/%s.jpg' % name))
            gt = np.array(Image.open('/data1/zhang_runtong/datasets/VSAC/novel/mask/%s.png' % name))
            pred[pred != 0] = 1

            if len(img.shape) == 2:
                img = np.expand_dims(img, 2).repeat(3, axis=2)
            image = img * pred[:, :, None]
            # image = img
            image = Image.fromarray(image)
            image = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                torch.cuda.synchronize()
                logits_per_image, logits_per_text = CLIP(image, text)

            torch.cuda.synchronize()

            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            pred_cat_idx = np.argmax(probs) + 1

            pred = pred * int(pred_cat_idx)
            pred_list.append(pred)
            gt_list.append(gt)

        tools.compute_miou_classes(gt_list, pred_list, n_class=17)

