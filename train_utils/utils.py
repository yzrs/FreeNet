import logging
import math
import os
import re
from collections import OrderedDict
import random

import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import json
from train_utils import transforms
from torch.utils.data import DataLoader

from train_utils.dataset import CocoKeypoint
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from models.hrnet import HighResolutionNet
from train_utils.transforms import get_max_preds

os.environ['MPLBACKEND'] = 'TkAgg'
logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AvgImgMSELoss(object):
    def __init__(self,kps_weights,num_joints):
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.kps_weights = kps_weights
        self.num_joints = num_joints

    def __call__(self, logits, targets, visible):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        # device = logits.device
        logits = logits.to(torch.float32)
        kps_weights = np.array(self.kps_weights,dtype=np.float32).reshape((self.num_joints,))

        bs = logits.shape[0]
        kps_weights = np.repeat(kps_weights, bs)
        kps_weights = kps_weights.reshape((bs, self.num_joints))
        kps_weights = torch.from_numpy(kps_weights)

        assert kps_weights.shape == visible.shape, 'kps_weights and visible should have the same shape'
        kps_weights[visible == 0] = 0
        # kps_weights = kps_weights.to(device)
        kps_weights = kps_weights.cuda()

        loss = self.criterion(logits, targets).mean(dim=[2, 3])
        loss = torch.sum(loss * kps_weights) / bs
        return loss


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        """根据step数返回一个学习率倍率因子"""
        if x >= warmup_iters:  # 当迭代数大于给定的warmup_iters时，倍率因子为1
            return 1
        alpha = float(x) / warmup_iters
        # 迭代过程中倍率因子从warmup_factor -> 1
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def load_img_show(images, targets):
    # image show
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images_tensor = images * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    images_np = images_tensor.numpy()
    for i, img_np in enumerate(images_np):
        print(targets[i]['image_path'])
        plt.imshow(img_np.transpose(1, 2, 0))
        plt.show()


def collate_fn(batch):
    imgs_tuple, targets_tuple = tuple(zip(*batch))
    imgs_tensor = torch.stack(imgs_tuple)
    return imgs_tensor, targets_tuple







def convert_output_to_coco_format_batch_mix(output, target, num_joints=17):
    # 假设您已经将模型的输出转换为形状为 (17, 2) 和 (17, 1) 的张量
    keypoints = output[0].reshape(num_joints, 2)  # 关键点坐标，形状为 (17, 2)
    visibilities = output[1].reshape(num_joints, 1)  # 关键点可见性，形状为 (17, 1)

    scores = output[1]
    mask = np.greater(scores, 0.2)
    if mask.sum() == 0:
        k_score = 0
    else:
        k_score = np.mean(scores[mask])
    # 将关键点坐标和可见性拼接在一起
    keypoints = np.concatenate([keypoints, visibilities], axis=-1)

    # 将关键点坐标和可见性重塑为一维数组
    keypoints = keypoints.reshape(-1).tolist()

    # 创建 COCO 数据集格式的结果
    result = {
        'image_id': target['image_id'],
        'category_id': target['category_id'],
        'bbox': target['box'],
        'anno_id': target['anno_id'],
        'keypoints': keypoints,
        'score': target['score'] * k_score,
        'dataset': target['dataset'],
        'mode': target['mode']
    }
    return result


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def generate_heatmap(coords,hard_pseudo_visible,heatmap_size=(64,64), sigma=2.0):
    batch_size, num_joints, _ = coords.shape

    heatmaps = torch.zeros((batch_size, num_joints, heatmap_size[0], heatmap_size[1]), dtype=torch.float32)

    for i in range(batch_size):
        for j in range(num_joints):
            # if this keypoint is not visible,just skip
            if hard_pseudo_visible[i][j] == 0:
                continue
            coord = coords[i, j].cpu()
            x, y = coord[0], coord[1]

            ul = [int(max(x - 3 * sigma, 0)), int(max(y - 3 * sigma, 0))]
            br = [int(min(x + 3 * sigma, heatmap_size[1] - 1)), int(min(y + 3 * sigma, heatmap_size[0] - 1))]

            # Check if the keypoint is within heatmap boundaries
            if ul[0] > heatmap_size[1] - 1 or ul[1] > heatmap_size[0] - 1 or br[0] < 0 or br[1] < 0:
                continue

            img_x = torch.arange(heatmap_size[1], dtype=torch.float32).unsqueeze(0).repeat(heatmap_size[0], 1)
            img_y = torch.arange(heatmap_size[0], dtype=torch.float32).unsqueeze(1).repeat(1, heatmap_size[1])

            heatmap = torch.exp(-((img_x - x) ** 2 + (img_y - y) ** 2) / (2 * sigma ** 2))

            # Normalize the heatmap values between 0 and 1
            heatmap = heatmap / heatmap.max().item()

            heatmaps[i, j] = heatmap

    return heatmaps


def get_block_list(args,level):
    kps_a = ['L_F_hip','L_F_knee','L_F_paw','R_F_hip','R_F_knee','R_F_paw']
    kps_b = ['R_B_hip',  'R_B_knee','R_B_paw','L_B_hip','L_B_knee', 'L_B_paw']
    common_kps_small = ['R_eye', 'neck','L_eye', 'nose', 'tail']
    common_kps_middle = ['R_eye', 'neck','L_eye', 'nose', 'tail','L_F_paw','R_F_paw','L_B_paw','R_B_paw']
    common_kps_large = ['R_eye', 'neck','L_eye', 'nose', 'tail','L_F_paw','R_F_paw','L_B_paw','R_B_paw',
                        'L_F_hip','R_F_hip','L_B_hip','R_B_hip']

    with open(args.keypoints_path, 'r') as f:
        animal_kps_info = json.load(f)
    kps_definition = animal_kps_info['keypoints']
    kps_a_index = []
    kps_b_index = []
    common_kps_small_index = []
    common_kps_middle_index = []
    common_kps_large_index = []

    for kp_index, kp in enumerate(kps_definition):
        if kp in kps_a:
            kps_a_index.append(kp_index)
        if kp in kps_b:
            kps_b_index.append(kp_index)
        if kp in common_kps_small:
            common_kps_small_index.append(kp_index)
        if kp in common_kps_middle:
            common_kps_middle_index.append(kp_index)
        if kp in common_kps_large:
            common_kps_large_index.append(kp_index)

    common_kps_index = [common_kps_small_index,common_kps_middle_index,common_kps_large_index]

    index_a = []
    index_b = []
    block_index_a = []
    block_index_b = []
    for i in range(17):
        if i in kps_a_index or i in common_kps_index[level]:
            index_a.append(i)
        if i in kps_b_index or i in common_kps_index[level]:
            index_b.append(i)
        if i not in kps_a_index and i not in common_kps_index[level]:
            block_index_a.append(i)
        if i not in kps_b_index and i not in common_kps_index[level]:
            block_index_b.append(i)

    return block_index_a,block_index_b


def cosine_rampdown(current, rampdown_length):
    # assert 0 <= current <= rampdown_length
    current = np.clip(current, 0.0, rampdown_length)
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def get_current_topkrate(epoch, rampdown_epoch, min_rate):
    r = cosine_rampdown(epoch, rampdown_epoch)
    return np.clip(r, min_rate, 1)


def calculate_weight(start_step, mid_step, end_step, weight, current_step):
    if current_step < start_step:
        return 0.0
    elif current_step < mid_step:
        return weight * (current_step - start_step) / (mid_step - start_step)
    elif current_step < end_step:
        cur_step = current_step - mid_step
        rest_step = end_step - mid_step
        cur_step = np.clip(cur_step,0.0,rest_step)
        return float(.5 * (np.cos(np.pi * cur_step / rest_step) + 1)) * weight
    else:
        return weight


