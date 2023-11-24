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


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def module_load_state_dict(model, state_dict):
    try:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f'module.{k}'  # add `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def model_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        module_load_state_dict(model, state_dict)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
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


class KpLoss(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        # [num_kps] -> [B, num_kps]
        kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])

        # [B, num_kps, H, W] -> [B, num_kps]
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        loss = torch.sum(loss * kps_weights) / bs
        return loss


class KpLossLabel(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets, visible, args):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        logits = logits.to(torch.float32)
        key_info_path = args.keypoints_path
        num_joints = args.num_joints
        with open(key_info_path, "r") as f:
            animal_kps_info = json.load(f)
        kps_weights = np.array(animal_kps_info["kps_weights"],
                               dtype=np.float32).reshape((num_joints,))

        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        # [num_kps] -> [B, num_kps]
        # 使用 repeat 函数重复数组
        kps_weights = np.repeat(kps_weights, bs)
        # 使用 reshape 函数更改形状
        # [B, num_kps, H, W] -> [B, num_kps]
        kps_weights = kps_weights.reshape((bs, num_joints))
        # 使用 torch.from_numpy 将 numpy 数组转换为 CPU 张量
        # 使用 Tensor.to 将张量移动到 GPU 上
        kps_weights = torch.from_numpy(kps_weights)
        # print("=====================loss compute function==========================")
        # print("kps weights.shape:",kps_weights.shape)
        # print("visible.shape:",visible.shape)
        assert kps_weights.shape == visible.shape, 'kps_weights and visible should have the same shape'
        kps_weights[visible == 0] = 0
        kps_weights = kps_weights.to(device)

        loss = self.criterion(logits, targets).mean(dim=[2, 3])
        loss = torch.sum(loss * kps_weights) / bs
        return loss


class KpLossLabel_v2(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets, visible, args,indices=None):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        logits = logits.to(torch.float32)
        key_info_path = args.keypoints_path
        num_joints = args.num_joints
        with open(key_info_path, "r") as f:
            animal_kps_info = json.load(f)
        kps_weights = np.array(animal_kps_info["kps_weights"],
                               dtype=np.float32).reshape((num_joints,))

        if indices is not None:
            kps_weights = kps_weights[indices]
            num_joints = len(indices)

        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        # [num_kps] -> [B, num_kps]
        # 使用 repeat 函数重复数组
        kps_weights = np.repeat(kps_weights, bs)
        # 使用 reshape 函数更改形状
        # [B, num_kps, H, W] -> [B, num_kps]
        kps_weights = kps_weights.reshape((bs, num_joints))
        # 使用 torch.from_numpy 将 numpy 数组转换为 CPU 张量
        # 使用 Tensor.to 将张量移动到 GPU 上
        kps_weights = torch.from_numpy(kps_weights)
        # print("=====================loss compute function==========================")
        # print("kps weights.shape:",kps_weights.shape)
        # print("visible.shape:",visible.shape)
        assert kps_weights.shape == visible.shape, 'kps_weights and visible should have the same shape'
        kps_weights[visible == 0] = 0
        kps_weights = kps_weights.to(device)

        loss = self.criterion(logits, targets).mean(dim=[2, 3])
        loss = torch.sum(loss * kps_weights) / (bs * num_joints)
        return loss


class KpLossLabel_v3(object):
    def __init__(self,kps_weights,num_joints):
        self.kps_weights = kps_weights
        self.num_joints = num_joints
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets, visible, indices=None):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        logits = logits.to(torch.float32)
        kps_weights = np.array(self.kps_weights,dtype=np.float32).reshape((self.num_joints,))
        num_joints = self.num_joints

        if indices is not None:
            kps_weights = kps_weights[indices]
            num_joints = len(indices)

        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        # [num_kps] -> [B, num_kps]
        # 使用 repeat 函数重复数组
        kps_weights = np.repeat(kps_weights, bs)
        # 使用 reshape 函数更改形状
        # [B, num_kps, H, W] -> [B, num_kps]
        kps_weights = kps_weights.reshape((bs, num_joints))
        # 使用 torch.from_numpy 将 numpy 数组转换为 CPU 张量
        # 使用 Tensor.to 将张量移动到 GPU 上
        kps_weights = torch.from_numpy(kps_weights)
        # print("=====================loss compute function==========================")
        # print("kps weights.shape:",kps_weights.shape)
        # print("visible.shape:",visible.shape)
        assert kps_weights.shape == visible.shape, 'kps_weights and visible should have the same shape'
        kps_weights[visible == 0] = 0
        kps_weights = kps_weights.to(device)

        loss = self.criterion(logits, targets).mean(dim=[2, 3])
        loss = torch.sum(loss * kps_weights) / bs
        return loss


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
        # [num_kps, H, W] -> [B, num_kps, H, W]
        # [num_kps] -> [B, num_kps]
        # 使用 repeat 函数重复数组
        kps_weights = np.repeat(kps_weights, bs)
        # 使用 reshape 函数更改形状
        # [B, num_kps, H, W] -> [B, num_kps]
        kps_weights = kps_weights.reshape((bs, self.num_joints))
        # 使用 torch.from_numpy 将 numpy 数组转换为 CPU 张量
        # 使用 Tensor.to 将张量移动到 GPU 上
        kps_weights = torch.from_numpy(kps_weights)

        assert kps_weights.shape == visible.shape, 'kps_weights and visible should have the same shape'
        kps_weights[visible == 0] = 0
        # kps_weights = kps_weights.to(device)
        kps_weights = kps_weights.cuda()

        loss = self.criterion(logits, targets).mean(dim=[2, 3])
        loss = torch.sum(loss * kps_weights) / bs
        return loss


class AvgImgMSELoss_v2(object):
    def __init__(self,kps_weights,num_joints):
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.kps_weights = kps_weights
        self.num_joints = num_joints

    def __call__(self, logits, targets, visible,indices):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        # device = logits.device
        logits = logits.to(torch.float32)
        kps_weights = np.array(self.kps_weights,dtype=np.float32).reshape((self.num_joints,))

        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        # [num_kps] -> [B, num_kps]
        # 使用 repeat 函数重复数组
        kps_weights = np.repeat(kps_weights, bs)
        # 使用 reshape 函数更改形状
        # [B, num_kps, H, W] -> [B, num_kps]
        kps_weights = kps_weights.reshape((bs, self.num_joints))
        # 使用 torch.from_numpy 将 numpy 数组转换为 CPU 张量
        # 使用 Tensor.to 将张量移动到 GPU 上
        kps_weights = torch.from_numpy(kps_weights)

        assert kps_weights.shape == visible.shape, 'kps_weights and visible should have the same shape'
        kps_weights[visible == 0] = 0
        # kps_weights = kps_weights.to(device)
        kps_weights = kps_weights.cuda()

        loss = self.criterion(logits, targets).mean(dim=[2, 3])

        visible_target_kps_num = torch.sum(visible[:, indices], dim=1)
        avg_vis_kps_loss_per_img = torch.sum((loss * kps_weights)[:,indices],dim=1) / visible_target_kps_num
        return avg_vis_kps_loss_per_img


# 计算数据集中所有有双眼标签的动物的双眼间距的均值
def avg_eye_dist(args):
    eye_dists = []

    # 加载 COCO ground truth 数据
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset="ap_10k", mode="val", transform=data_transform["test"],
                          fixed_size=args.fixed_size, data_type="keypoints")
    test_dataset_loader = DataLoader(cocoGt,
                                     batch_size=8,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=args.workers,
                                     collate_fn=cocoGt.collate_fn)

    for _, targets in test_dataset_loader:
        for target in targets:
            keypoints = target['keypoints']
            visible = target['visible']
            if visible[0] > 0 and visible[1] > 0:
                eye_dist = np.linalg.norm(keypoints[0, :2] - keypoints[1, :2])
                eye_dists.append(eye_dist)
    avg_dist = np.mean(eye_dists)
    return avg_dist


def draw_keypoints(imgs, pred_keypoints, target_keypoints, index):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    imgs = imgs.permute(0, 2, 3, 1).numpy()  # 转换为numpy数组并调整维度顺序

    pred = pred_keypoints.numpy()  # 转换为numpy数组
    label = target_keypoints.numpy()  # 转换为numpy数组

    fig, axes = plt.subplots(1, 2, figsize=(6, 8))
    axes[0].imshow(imgs[index])
    for j, kp in enumerate(pred[index]):
        x, y = kp
        axes[0].scatter(x, y, c='r', s=8)
        axes[0].text(x, y, str(j), color='r', fontsize=8)
    axes[1].imshow(imgs[index])
    for j, kp in enumerate(label[index]):
        x, y = kp
        axes[1].scatter(x, y, c='b', s=8)
        axes[1].text(x, y, str(j), color='b', fontsize=8)

    plt.show()


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        """根据step数返回一个学习率倍率因子"""
        if x >= warmup_iters:  # 当迭代数大于给定的warmup_iters时，倍率因子为1
            return 1
        alpha = float(x) / warmup_iters
        # 迭代过程中倍率因子从warmup_factor -> 1
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def check_gt_points(imgs,targets):
    color = ['red']*17
    with open('info/ap_10k_keypoints_format.json','r') as f:
        kps_name = json.load(f)['keypoints']
    for index,target in enumerate(targets):
        img = imgs[index]
        # image show
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        images_tensor = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        images_np = images_tensor.numpy()

        kps = target['keypoints']
        vis = target['visible']
        valid_kps_x = []
        valid_kps_y = []
        for vi_index,vi in enumerate(vis):
            if vi > 0:
                valid_kps_x.append(kps[vi_index][0])
                valid_kps_y.append(kps[vi_index][1])
            else:
                valid_kps_x.append(0.0)
                valid_kps_y.append(0.0)
                # valid_kps_x.append(kps[vi_index][0])
                # valid_kps_y.append(kps[vi_index][1])

        print('===============')
        print("anno_id: ",target['anno_id'],", img_path: ",target['image_path'])
        for i in range(17):
            print(kps_name[i],vis[i],valid_kps_x[i],valid_kps_y[i])
        # print(vis[14],valid_kps_x[14],valid_kps_y[14])
        # print(vis[17],valid_kps_x[17],valid_kps_y[17])

        plt.scatter(valid_kps_x,valid_kps_y,c=color)
        plt.imshow(images_np.transpose(1, 2, 0))
        plt.show()


# for mean teacher debug
def check_mt_heatmap(select_num,imgs_tensor,heatmaps_tensor,visible=None):
    import matplotlib.pyplot as plt
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if visible is None:
        test_coords, hard_pseudo_visible = get_max_preds(heatmaps_tensor)
        hard_pseudo_visible = (hard_pseudo_visible > 0.4).float().squeeze(-1)
    else:
        hard_pseudo_visible = visible
        test_coords, _ = get_max_preds(heatmaps_tensor)

    test_coords_scale = (test_coords.cpu() * 4).numpy()
    sum_tensor = torch.sum(heatmaps_tensor, dim=1).cpu()
    for i in range(select_num):
        img = imgs_tensor[i].cpu()
        img_tensor = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        img_np = img_tensor.numpy()
        cor_x = test_coords_scale[i, :, 0]
        cor_y = test_coords_scale[i, :, 1]
        color = ['blue'] * 17
        size = [20] * 17
        for j, vis in enumerate(hard_pseudo_visible[i]):
            if vis == 0:
                color[j] = 'red'
                size[j] = 10
        plt.subplot(1, 2, 1)
        plt.imshow(img_np.transpose(1, 2, 0))
        plt.scatter(cor_x, cor_y, s=size, color=color)
        for j, (x, y) in enumerate(zip(cor_x, cor_y)):
            plt.text(x, y, str(j), color='white')  # 在对应位置上添加编号
        plt.subplot(1, 2, 2)
        plt.imshow(sum_tensor[i], cmap='hot')
        plt.colorbar()
        # 将标题添加到整个图像的上方
        plt.suptitle(f"img:{i} vis:{torch.sum(hard_pseudo_visible[i]).item()}")
        plt.tight_layout()
        plt.show()


def check_contrast(select_num,imgs_tensor_before,heatmaps_tensor_before,visible_before,
                   imgs_tensor_after,heatmaps_tensor_after,visible_after):
    import matplotlib.pyplot as plt
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_coords_before, _ = get_max_preds(heatmaps_tensor_before)
    test_coords_after, _ = get_max_preds(heatmaps_tensor_after)

    test_coords_scale_before = (test_coords_before.cpu() * 4).numpy()
    test_coords_scale_after = (test_coords_after.cpu() * 4).numpy()
    sum_tensor_before = torch.sum(heatmaps_tensor_before, dim=1).cpu()
    sum_tensor_after = torch.sum(heatmaps_tensor_after, dim=1).cpu()
    for i in range(select_num):
        img_before = imgs_tensor_before[i].cpu()
        img_tensor_before = img_before * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        img_np_before = img_tensor_before.numpy()
        cor_x_before = test_coords_scale_before[i, :, 0]
        cor_y_before = test_coords_scale_before[i, :, 1]
        color = ['blue'] * 17
        size = [20] * 17
        for j, vis in enumerate(visible_before[i]):
            if vis == 0:
                color[j] = 'red'
                size[j] = 10
        plt.subplot(2, 2, 1)
        plt.imshow(img_np_before.transpose(1, 2, 0))
        plt.scatter(cor_x_before, cor_y_before, s=size, color=color)
        for j, (x, y) in enumerate(zip(cor_x_before, cor_y_before)):
            plt.text(x, y, str(j), color='white')  # 在对应位置上添加编号
        plt.subplot(2, 2, 2)
        plt.imshow(sum_tensor_before[i], cmap='hot')
        plt.colorbar()

        img_after = imgs_tensor_after[i].cpu()
        img_tensor_after = img_after * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        img_np_after = img_tensor_after.numpy()
        cor_x_after = test_coords_scale_after[i, :, 0]
        cor_y_after = test_coords_scale_after[i, :, 1]
        color = ['blue'] * 17
        size = [20] * 17
        for j, vis in enumerate(visible_after[i]):
            if vis == 0:
                color[j] = 'red'
                size[j] = 10
        plt.subplot(2, 2, 3)
        plt.imshow(img_np_after.transpose(1, 2, 0))
        plt.scatter(cor_x_after, cor_y_after, s=size, color=color)
        for j, (x, y) in enumerate(zip(cor_x_after, cor_y_after)):
            plt.text(x, y, str(j), color='white')  # 在对应位置上添加编号
        plt.subplot(2, 2, 4)
        plt.imshow(sum_tensor_after[i].detach(), cmap='hot')
        plt.colorbar()
        # 将标题添加到整个图像的上方
        plt.suptitle(f"img:{i} vis:{torch.sum(visible_after[i]).item()}")
        plt.tight_layout()
        plt.show()


def train_one_epoch(args, model, optimizer, data_loader, device, epoch, warmup=False, scaler=None):
    losses = AverageMeter()
    now_lr = 0.0

    lr_scheduler = None
    if epoch < 3 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 100
        warmup_iters = min(100, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    pbar = tqdm(range(len(data_loader)))

    for imgs, targets in data_loader:
        imgs = torch.stack([img.to(device) for img in imgs])
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)

        target_heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        target_visible = torch.stack([torch.tensor(t["visible"]) for t in targets])
        criterion = KpLossLabel()
        loss = criterion(logits, target_heatmaps, target_visible, args)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        losses.update(loss.item())
        now_lr = optimizer.param_groups[0]["lr"]
        pbar.set_description(f"Epoch:{epoch}/{args.epochs}, Train Loss: {losses.avg:.7f}. Training")
        pbar.update()

    pbar.close()
    train_path = os.path.join(args.output_dir, "./info/train_log.txt")
    with open(train_path, "a") as f:
        f.write(f"epoch:{epoch},train Loss: {losses.avg:.6f}\n")

    return losses.avg, now_lr


def train_one_epoch_parallel(args, model, optimizer, data_loader, device, epoch, warmup=False, scaler=None):
    losses = AverageMeter()
    now_lr = 0.0

    lr_scheduler = None
    if epoch < 3 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 100
        warmup_iters = min(100, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    pbar = tqdm(range(len(data_loader)))

    with open(args.keypoints_path,'r') as f:
        animal_kps_info = json.load(f)
    criterion = AvgImgMSELoss(kps_weights=animal_kps_info['kps_weights'], num_joints=args.num_joints)

    for imgs, targets in data_loader:
        imgs = torch.stack([img for img in imgs]).cuda()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)

        target_heatmaps = torch.stack([t["heatmap"] for t in targets]).cuda()
        target_visible = torch.stack([torch.tensor(t["visible"]) for t in targets])
        loss = criterion(logits, target_heatmaps, target_visible)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        losses.update(loss.item())
        now_lr = optimizer.param_groups[0]["lr"]
        pbar.set_description(f"Epoch:{epoch}/{args.epochs}, Train Loss: {losses.avg:.7f}. Training")
        pbar.update()

    pbar.close()
    train_path = os.path.join(args.output_dir, "./info/train_log.txt")
    with open(train_path, "a") as f:
        f.write(f"epoch:{epoch},train Loss: {losses.avg:.6f}\n")

    return losses.avg, now_lr


def train_one_epoch_ema(args, model, ema_model,optimizer, data_loader, device, epoch, warmup=False, scaler=None):
    # with open(args.keypoints_path, "r") as f:
    #     animal_kps_info = json.load(f)

    losses = AverageMeter()
    now_lr = 0.0

    lr_scheduler = None
    if epoch < 3 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 100
        warmup_iters = min(100, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    pbar = tqdm(range(len(data_loader)))
    for imgs, targets in data_loader:
        # load_img_show(imgs,targets)

        # check_gt_points(imgs,targets)

        imgs = torch.stack([img.to(device) for img in imgs])
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)

        target_heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        target_visible = torch.stack([torch.tensor(t["visible"]) for t in targets])
        criterion = KpLossLabel()
        loss = criterion(logits, target_heatmaps, target_visible, args)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        ema_model.update_parameters(model)
        losses.update(loss.item())
        now_lr = optimizer.param_groups[0]["lr"]
        pbar.set_description(f"Epoch:{epoch}/{args.epochs}, Train Loss: {losses.avg:.7f}. Training")
        pbar.update()

    pbar.close()
    train_path = os.path.join(args.output_dir, "./info/train_log.txt")
    with open(train_path, "a") as f:
        f.write(f"epoch:{epoch},train Loss: {losses.avg:.6f}\n")

    return losses.avg, now_lr


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


def load_img_show_ori_aug_comparison(images_ori,images_aug):
    # image show
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images_tensor_ori = images_ori * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    images_np_ori = images_tensor_ori.numpy()
    images_tensor_aug = images_aug * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    images_np_aug = images_tensor_aug.numpy()
    for i, imgs_np in enumerate(zip(images_np_ori,images_np_aug)):
        img_ori,img_aug = imgs_np
        plt.subplot(1,2,1)
        plt.imshow(img_ori.transpose(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(img_aug.transpose(1, 2, 0))
        plt.show()


def load_img_dt_show(images,targets):
    # image show
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images_tensor = images * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    images_np = images_tensor.numpy()
    for i, img_np in enumerate(images_np):
        coord = targets[i]['keypoints']
        coord_x = coord[:,0]
        coord_y = coord[:,1]
        color = ['blue']*17
        size = [25] * 17
        visible = targets[i]['visible']
        for vi_index,vi in enumerate(visible):
            if vi == 0:
                color[vi_index] = 'yellow'
                size[vi_index] = 10
        plt.imshow(img_np.transpose(1, 2, 0))
        plt.scatter(coord_x, coord_y, c=color, s=size)  # 在图像上绘制点，颜色为红色，大小为10
        for j, (x, y) in enumerate(zip(coord_x, coord_y)):
            plt.annotate(j, (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8,
                         color='white')  # 给每个坐标点添加标号
        plt.show()


# param: flag
# true for ap_10k
# false for animal_pose
def compute_oks(oks_list, oks_sum_list, vis_sum_list, gt, dt, threshold, kpt_oks_sigmas):
    vars = (np.array(kpt_oks_sigmas) * 2) ** 2
    k = len(kpt_oks_sigmas)
    # compute oks between each detection and ground truth object
    # create bounds for ignore regions(double the gt bbox)
    g = np.array(gt['keypoints'])
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    k1 = np.count_nonzero(vg > 0)
    bb = gt['bbox']
    x0 = bb[0] - bb[2]
    x1 = bb[0] + bb[2] * 2
    y0 = bb[1] - bb[3]
    y1 = bb[1] + bb[3] * 2
    if dt['anno_id'] == gt['id']:
        d = np.array(dt['keypoints'])
        xd = d[0::3]
        yd = d[1::3]
        if k1 > 0:
            # measure the per-keypoint distance if keypoints visible
            dx = xd - xg
            dy = yd - yg
        else:
            # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
            z = np.zeros((k))
            dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
            dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
        e = (dx ** 2 + dy ** 2) / vars / (gt['area'] + np.spacing(1)) / 2
        # e = (dx ** 2 + dy ** 2) / (2 * vars * gt['area'])
        e = np.exp(-e)

        # keypoint level threshold
        if k1 > 0:
            e = [e[ind] if vg[ind] > 0 and e[ind] > threshold else 0.0 for ind in range(k)]
            vg = [1 if vg[ind] > 0 else 0 for ind in range(len(vg))]

        for i, j, sublist in zip(e, vg, oks_list):
            if j != 0:
                sublist.append(i)
        oks_sum_list = [a + b for a, b in zip(oks_sum_list, e)]
        vis_sum_list = [a + b for a, b in zip(vis_sum_list, vg)]
    return oks_sum_list, vis_sum_list


def compute_oks_mix(oks_list, gt, dt, threshold, kpt_oks_sigmas):
    vars = (np.array(kpt_oks_sigmas) * 2) ** 2
    k = len(kpt_oks_sigmas)
    # compute oks between each detection and ground truth object
    # create bounds for ignore regions(double the gt bbox)
    g = np.array(gt['keypoints'])
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    k1 = np.count_nonzero(vg > 0)
    bb = gt['bbox']
    x0 = bb[0] - bb[2]
    x1 = bb[0] + bb[2] * 2
    y0 = bb[1] - bb[3]
    y1 = bb[1] + bb[3] * 2
    if dt['anno_id'] == gt['id']:
        d = np.array(dt['keypoints'])
        xd = d[0::3]
        yd = d[1::3]
        if k1 > 0:
            # measure the per-keypoint distance if keypoints visible
            dx = xd - xg
            dy = yd - yg
        else:
            # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
            z = np.zeros((k))
            dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
            dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
        e = (dx ** 2 + dy ** 2) / vars / (gt['area'] + np.spacing(1)) / 2
        # e = (dx ** 2 + dy ** 2) / (2 * vars * gt['area'])
        e = np.exp(-e)

        # keypoint level threshold
        if k1 > 0:
            e = [e[ind] if vg[ind] > 0 and e[ind] > threshold else 0.0 for ind in range(k)]
            vg = [1 if vg[ind] > 0 else 0 for ind in range(len(vg))]

        for i, j, sublist in zip(e, vg, oks_list):
            if j != 0:
                sublist.append(i)
    return oks_list


def compute_oks_no_threshold(oks_list, oks_sum_list, vis_sum_list, gt, dt, kpt_oks_sigmas):
    vars = (np.array(kpt_oks_sigmas) ** 2) * 2
    k = len(kpt_oks_sigmas)
    # compute oks between each detection and ground truth object
    # create bounds for ignore regions(double the gt bbox)
    g = np.array(gt['keypoints'])
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    k1 = np.count_nonzero(vg > 0)
    bb = gt['bbox']
    x0 = bb[0] - bb[2]
    x1 = bb[0] + bb[2] * 2
    y0 = bb[1] - bb[3]
    y1 = bb[1] + bb[3] * 2
    if dt['anno_id'] == gt['id']:
        d = np.array(dt['keypoints'])
        xd = d[0::3]
        yd = d[1::3]
        if k1 > 0:
            # measure the per-keypoint distance if keypoints visible
            dx = xd - xg
            dy = yd - yg
        else:
            # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
            z = np.zeros((k))
            dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
            dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
        e = (dx ** 2 + dy ** 2) / vars / (gt['area'] + np.spacing(1)) / 2
        # e = (dx ** 2 + dy ** 2) / (vars * gt['area'])
        e = np.exp(-e)

        # keypoint level threshold
        if k1 > 0:
            e = [e[ind] if vg[ind] > 0 else 0.0 for ind in range(k)]
            vg = [1 if vg[ind] > 0 else 0 for ind in range(len(vg))]

        for i, j, sublist in zip(e, vg, oks_list):
            if j != 0:
                sublist.append(i)
        oks_sum_list = [a + b for a, b in zip(oks_sum_list, e)]
        vis_sum_list = [a + b for a, b in zip(vis_sum_list, vg)]
    return oks_sum_list, vis_sum_list


def keypoints_show(args, model_name, dataset='ap_10k', mode='val'):
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode, transform=data_transform["test"],
                          fixed_size=args.fixed_size, data_type="keypoints")
    res_file = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_results.json'
    cocoDt = cocoGt.coco.loadRes(res_file)

    fig, axs = plt.subplots(3, 3)
    axs = axs.ravel()
    img_ids = random.sample(cocoGt.coco.getImgIds(), k=len(cocoGt.coco.getImgIds()))
    for i, img_id in enumerate(img_ids):
        # get the image
        img = cocoGt.coco.loadImgs(img_id)[0]
        mp_img = mpimg.imread(f'{data_root}/data/{img["file_name"]}')
        axs[i % 9].imshow(mp_img)

        # get the GT keypoints
        gt_ann_ids = cocoGt.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        gt_anns = cocoGt.coco.loadAnns(gt_ann_ids)

        # get the Pred keypoints
        dt_ann_ids = cocoDt.getAnnIds(imgIds=img['id'], iscrowd=None)
        dt_anns = cocoDt.loadAnns(dt_ann_ids)

        for gt_ann in gt_anns:
            for dt_ann in dt_anns:
                if gt_ann['id'] == dt_ann['anno_id']:
                    gt = gt_ann['keypoints']
                    dt = dt_ann['keypoints']
                    # reshape to (17,3)
                    # gt = np.array(gt).reshape(17,3)
                    # dt = np.array(dt).reshape(17,3)
                    # # neck keypoint index is 3 for ap-10k
                    # gt_neck = gt[3][:2]
                    # dt_neck = dt[3][:2]
                    gt_x = gt[0::3]
                    gt_y = gt[1::3]
                    dt_x = dt[0::3]
                    dt_y = dt[1::3]

                    colors = ['red'] * len(gt_x)  # 使用红色标记初始化所有数据点
                    colors[3] = 'green'  # 将第四个点的颜色设置为绿色

                    axs[i % 9].scatter(gt_x, gt_y, color=colors, s=10)
                    # axs[i % 9].scatter(dt_x,dt_y,color='red')
        if i % 9 == 8:
            manager = plt.get_current_fig_manager()
            manager.resize(*manager.window.maxsize())
            plt.show()
            fig, axs = plt.subplots(3, 3)
            axs = axs.ravel()


def keypoints_show_one_by_one(args, model_name, dataset='ap_10k', mode='val'):
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode, transform=data_transform["test"],
                          fixed_size=args.fixed_size, data_type="keypoints")
    res_file = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_results.json'
    cocoDt = cocoGt.coco.loadRes(res_file)

    img_ids = random.sample(cocoGt.coco.getImgIds(), k=len(cocoGt.coco.getImgIds()))
    for i, img_id in enumerate(img_ids):
        # 创建新的画布和子图
        fig, axs = plt.subplots(1, 1)

        # 取得图像
        img = cocoGt.coco.loadImgs(img_id)[0]
        mp_img = mpimg.imread(f'{data_root}/data/{img["file_name"]}')
        axs.imshow(mp_img)
        # 获取GT关键点
        gt_ann_ids = cocoGt.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        gt_anns = cocoGt.coco.loadAnns(gt_ann_ids)
        # 获取Pred关键点
        dt_ann_ids = cocoDt.getAnnIds(imgIds=img['id'], iscrowd=None)
        dt_anns = cocoDt.loadAnns(dt_ann_ids)
        for gt_ann in gt_anns:
            for dt_ann in dt_anns:
                if gt_ann['id'] == dt_ann['anno_id']:
                    gt = gt_ann['keypoints']
                    dt = dt_ann['keypoints']
                    gt_x = gt[0::3]
                    gt_y = gt[1::3]
                    dt_x = dt[0::3]
                    dt_y = dt[1::3]
                    colors = ['#ff0000'] * len(gt_x)
                    colors[3] = '#0000ff'
                    # 设置标点大小的数组
                    sizes = [100 if color == '#0000ff' else 40 for color in colors]
                    axs.scatter(gt_x, gt_y, color=colors, s=sizes)
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        plt.show()


def gt_keypoints_show_one_by_one(args, dataset='ap_10k', mode='val'):
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode, transform=data_transform["test"],
                          fixed_size=args.fixed_size, data_type="keypoints")

    img_ids = random.sample(cocoGt.coco.getImgIds(), k=len(cocoGt.coco.getImgIds()))
    for i, img_id in enumerate(img_ids):
        # 创建新的画布和子图
        fig, axs = plt.subplots(1, 1)

        # 取得图像
        img = cocoGt.coco.loadImgs(img_id)[0]
        mp_img = mpimg.imread(f'{data_root}/data/{img["file_name"]}')
        axs.imshow(mp_img)
        # 获取GT关键点
        gt_ann_ids = cocoGt.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        gt_anns = cocoGt.coco.loadAnns(gt_ann_ids)
        for gt_ann in gt_anns:
            gt = gt_ann['keypoints']
            gt_x = gt[0::3]
            gt_y = gt[1::3]
            colors = ['#ff0000'] * len(gt_x)
            colors[18] = '#0000ff'
            # 设置标点大小的数组
            sizes = [100 if color == '#0000ff' else 40 for color in colors]
            axs.scatter(gt_x, gt_y, color=colors, s=sizes)
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        plt.show()


def collate_fn(batch):
    imgs_tuple, targets_tuple = tuple(zip(*batch))
    imgs_tensor = torch.stack(imgs_tuple)
    return imgs_tensor, targets_tuple


def json_generate_by_name(args, name, dataset, path):
    device = args.device
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    test_dataset_loader = DataLoader(dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=args.workers,
                                     collate_fn=dataset.collate_fn)
    # 将模型输出转换为 COCO 关键点结果格式
    # create model
    model = HighResolutionNet(num_joints=args.num_joints)

    # 载入你自己训练好的模型权重
    weights_path = f"{args.output_dir}/save_weights/{name}.pth"
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    # model.load_state_dict(torch.load(weights_path, map_location='cpu')['teacher_model'])
    # model.load_state_dict(torch.load(weights_path, map_location='cpu')['student_model'])
    # model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    # print(model)
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for images, targets in tqdm(test_dataset_loader, desc="Model is predicting"):
            # 将图片传入指定设备device
            images = images.to(device)

            # inference
            outputs = model(images)

            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            result = convert_output_to_coco_format(outputs, targets, args.num_joints)
            results.append(result)

    with open(path, 'w') as f:
        json.dump(results, f)


def json_generate(args, model, dataset, path):
    device = args.device
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    test_dataset_loader = DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=args.workers,
                                     collate_fn=dataset.collate_fn)
    # 将模型输出转换为 COCO 关键点结果格式
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for images, targets in tqdm(test_dataset_loader, desc="Model is predicting for complete kps"):
            # 将图片传入指定设备device
            images = images.to(device)
            # inference
            outputs = model(images)
            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            for output, target in zip(zip(outputs[0], outputs[1]), targets):
                result = convert_output_to_coco_format_batch(output, target, num_joints=args.num_joints)
                results.append(result)

    with open(path, 'w') as f:
        json.dump(results, f)


def json_generate_batch(args, name, dataset, path):
    device = args.device
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    test_dataset_loader = DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=args.workers,
                                     collate_fn=collate_fn)
    # 将模型输出转换为 COCO 关键点结果格式
    # create model
    model = HighResolutionNet(num_joints=args.num_joints)

    # from ScarceNet.lib.models.pose_hrnet_part import get_pose_net
    # get_pose_net()

    # 载入你自己训练好的模型权重
    weights_path = f"{args.output_dir}/save_weights/{name}.pth"
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for images, targets in tqdm(test_dataset_loader, desc="Model is predicting for complete kps"):
            # 将图片传入指定设备device
            images = images.to(device)
            # inference
            outputs = model(images)
            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            for output, target in zip(zip(outputs[0], outputs[1]), targets):
                result = convert_output_to_coco_format_batch(output, target, num_joints=args.num_joints)
                results.append(result)

    with open(path, 'w') as f:
        json.dump(results, f)


def json_generate_key_batch(args, name, dataset, key, path):
    device = args.device
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    test_dataset_loader = DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=args.workers,
                                     collate_fn=collate_fn)
    # 将模型输出转换为 COCO 关键点结果格式
    # create model
    model = HighResolutionNet(num_joints=args.num_joints)

    # 载入你自己训练好的模型权重
    weights_path = f"{args.output_dir}/save_weights/{name}.pth"
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint[key])
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for images, targets in tqdm(test_dataset_loader, desc="Model is predicting for complete kps"):
            # 将图片传入指定设备device
            images = images.to(device)
            # inference
            outputs = model(images)
            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            for output, target in zip(zip(outputs[0], outputs[1]), targets):
                result = convert_output_to_coco_format_batch(output, target, num_joints=args.num_joints)
                results.append(result)

    with open(path, 'w') as f:
        json.dump(results, f)


def json_generate_batch_no_flip(args, name, dataset, path):
    device = args.device
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    test_dataset_loader = DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=args.workers,
                                     collate_fn=collate_fn)
    # 将模型输出转换为 COCO 关键点结果格式
    # create model
    model = HighResolutionNet(num_joints=args.num_joints)

    # 载入你自己训练好的模型权重
    weights_path = f"{args.output_dir}/save_weights/{name}.pth"
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    # print(model)
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for images, targets in tqdm(test_dataset_loader, desc="Model is predicting for complete kps"):
            # 将图片传入指定设备device
            images = images.to(device)
            # inference
            outputs = model(images)

            # flipped_images = transforms.flip_images(images)
            # flipped_outputs = model(flipped_images)
            # flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            # # feature is not aligned, shift flipped heatmap for higher accuracy
            # # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            # flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            # outputs = (outputs + flipped_outputs) * 0.5

            # outputs = transforms.flip_back(outputs,animal_kps_info["flip_pairs"])

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            for output, target in zip(zip(outputs[0], outputs[1]), targets):
                result = convert_output_to_coco_format_batch(output, target, num_joints=args.num_joints)
                results.append(result)

    with open(path, 'w') as f:
        json.dump(results, f)


# 需要使用独立的collate_fn函数
def json_generate_batch_mix(args, name, dataset, path,num_joints=26):
    device = args.device
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    test_dataset_loader = DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=args.workers,
                                     collate_fn=collate_fn)
    # 将模型输出转换为 COCO 关键点结果格式
    # create model
    model = HighResolutionNet(num_joints=args.num_joints)

    # 载入你自己训练好的模型权重
    weights_path = f"{args.output_dir}/save_weights/{name}.pth"
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    # model.load_state_dict(torch.load(weights_path, map_location='cpu')['teacher_model'])
    # model.load_state_dict(torch.load(weights_path, map_location='cpu')['student_model'])
    # model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    # model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['state_dict'])
    # print(model)
    model.to(device)
    model.eval()

    results = []
    if num_joints == 26:
        trans = transforms.OnlyLabelFormatTrans(extend_flag=False)
    elif num_joints == 21:
        trans = transforms.OnlyLabelFormatTransAP10KAnimalPose(extend_flag=False)
    else:
        return

    with torch.no_grad():
        for images, targets in tqdm(test_dataset_loader, desc="Model is predicting for specific kps"):
            # 将图片传入指定设备device
            images = images.to(device)
            # inference
            outputs = model(images)

            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            for output, target in zip(zip(outputs[0], outputs[1]), targets):
                result = convert_output_to_coco_format_batch_mix(output, target, num_joints=args.num_joints)
                result = trans(result)
                results.append(result)

    with open(path, 'w') as f:
        json.dump(results, f)


def json_generate_batch_mix_key(args, name, dataset, path,key="model",num_joints=26):
    """
        for single GPU
        if batch_size is too small
        this will be faster than running on multi GPUs
        :param args:
        :param name: weights id like model-0
        :param dataset: val_dataset used to create dataloader
        :param path: save path of prediction json file
        :param key: model key of checkpoint like "teacher_model" or "student_model" or "model"
        :return: None
    """
    device = args.device
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    test_dataset_loader = DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=args.workers,
                                     collate_fn=collate_fn)
    # 将模型输出转换为 COCO 关键点结果格式
    # create model
    model = HighResolutionNet(num_joints=args.num_joints)

    # 载入你自己训练好的模型权重
    weights_path = f"{args.output_dir}/save_weights/{name}.pth"
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')[key])
    # print(model)
    model.to(device)
    model.eval()

    results = []

    if num_joints == 26:
        trans = transforms.OnlyLabelFormatTrans(extend_flag=False)
    elif num_joints == 21:
        trans = transforms.OnlyLabelFormatTransAP10KAnimalPose(extend_flag=False)
    else:
        return

    with torch.no_grad():
        for images, targets in tqdm(test_dataset_loader, desc="Model is predicting for specific kps"):
            # 将图片传入指定设备device
            images = images.to(device)
            # inference
            outputs = model(images)

            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            for output, target in zip(zip(outputs[0], outputs[1]), targets):
                result = convert_output_to_coco_format_batch_mix(output, target, num_joints=args.num_joints)
                result = trans(result)
                results.append(result)

    with open(path, 'w') as f:
        json.dump(results, f)


def json_generate_batch_mix_key_parallel(args, name, dataset, path,key="model"):
    """
        for multiGPUs
        batch_size = batch_size * (1 + args.mu)
        if batch_size is too small
        this will be slower than running on single GPU
        In this func,we need to load weights from checkpoints,which takes a little more time
        So we have another function  json_generate_batch_mix_model_parallel below , to save this time-spending
        :param args:
        :param name: weights id like model-0
        :param dataset: val_dataset used to create dataloader
        :param path: save path of prediction json file
        :param key: model key of checkpoint like "teacher_model" or "student_model" or "model"
        :return: None
    """
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    test_dataset_loader = DataLoader(dataset,
                                     batch_size=args.batch_size * (1 + args.mu),
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=args.workers,
                                     collate_fn=collate_fn)
    # 将模型输出转换为 COCO 关键点结果格式
    # create model
    model = HighResolutionNet(num_joints=args.num_joints)

    # 载入你自己训练好的模型权重
    weights_path = f"{args.output_dir}/save_weights/{name}.pth"
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')[key])
    # print(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    model.eval()

    results = []

    with torch.no_grad():
        for images, targets in tqdm(test_dataset_loader, desc="Model is predicting for specific kps"):
            # 将图片传入指定设备device
            images = images.cuda()
            # inference
            outputs = model(images)

            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            for output, target in zip(zip(outputs[0], outputs[1]), targets):
                result = convert_output_to_coco_format_batch_mix(output, target, num_joints=args.num_joints)
                trans = transforms.OnlyLabelFormatTrans(extend_flag=False)
                result = trans(result)
                results.append(result)

    with open(path, 'w') as f:
        json.dump(results, f)


def json_generate_batch_mix_model_parallel(args, model, dataset, path,num_joints=26):
    """
        for multiGPUs
        batch_size = batch_size * (1 + args.mu)
        if batch_size is too small
        this will be slower than running on single GPU
        :param args:
        :param model: model used for validating
        :param dataset: val_dataset used to create dataloader
        :param path: save path of prediction json file
        :return: None
    """
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    test_dataset_loader = DataLoader(dataset,
                                     batch_size=args.batch_size * (1 + args.mu),
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=args.workers,
                                     collate_fn=collate_fn)
    model.eval()
    results = []

    if num_joints == 26:
        trans = transforms.OnlyLabelFormatTrans(extend_flag=False)
    else:
        trans = transforms.OnlyLabelFormatTransAP10KAnimalPose(extend_flag=False)

    with torch.no_grad():
        for images, targets in tqdm(test_dataset_loader, desc="Model is predicting for specific kps"):
            # 将图片传入指定设备device
            images = images.cuda()
            # inference
            outputs = model(images)

            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            for output, target in zip(zip(outputs[0], outputs[1]), targets):
                result = convert_output_to_coco_format_batch_mix(output, target, num_joints=args.num_joints)
                result = trans(result)
                results.append(result)

    with open(path, 'w') as f:
        json.dump(results, f)


def json_generate_batch_ssl(args, name, dataset, path,student=True):
    device = args.device
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    test_dataset_loader = DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=args.workers,
                                     collate_fn=collate_fn)
    # 将模型输出转换为 COCO 关键点结果格式
    # create model
    model = HighResolutionNet(num_joints=args.num_joints)

    # 载入你自己训练好的模型权重
    weights_path = f"{args.output_dir}/save_weights/{name}.pth"
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    if student:
        model.load_state_dict(torch.load(weights_path, map_location='cpu')['student_model'])
    else:
        model.load_state_dict(torch.load(weights_path, map_location='cpu')['teacher_model'])
    # print(model)
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for images, targets in tqdm(test_dataset_loader, desc="Model is predicting for complete kps"):
            # 将图片传入指定设备device
            images = images.to(device)
            # inference
            outputs = model(images)

            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            for output, target in zip(zip(outputs[0], outputs[1]), targets):
                result = convert_output_to_coco_format_batch(output, target, num_joints=args.num_joints)
                results.append(result)

    with open(path, 'w') as f:
        json.dump(results, f)


def json_generate_batch_ssl_v2(args, name, dataset, path,key_name):
    device = args.device
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    test_dataset_loader = DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=args.workers,
                                     collate_fn=collate_fn)
    # 将模型输出转换为 COCO 关键点结果格式
    # create model
    model = HighResolutionNet(num_joints=args.num_joints)

    # 载入你自己训练好的模型权重
    weights_path = f"{args.output_dir}/save_weights/{name}.pth"
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)

    model.load_state_dict(torch.load(weights_path, map_location='cpu')[key_name])
    # print(model)
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for images, targets in tqdm(test_dataset_loader, desc="Model is predicting for complete kps"):
            # 将图片传入指定设备device
            images = images.to(device)
            # inference
            outputs = model(images)

            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            for output, target in zip(zip(outputs[0], outputs[1]), targets):
                result = convert_output_to_coco_format_batch(output, target, num_joints=args.num_joints)
                results.append(result)

    with open(path, 'w') as f:
        json.dump(results, f)


def convert_output_to_coco_format(output, target, num_joints=17):
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
        'image_id': target[0]['image_id'],
        'category_id': target[0]['category_id'],
        'keypoints': keypoints,
        'score': target[0]['score'] * k_score
    }
    return result


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


def convert_output_to_coco_format_batch(output, target, num_joints=17):
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
        'score': target['score'] * k_score
    }
    return result


def compute_nme(true_keypoints, pred_keypoints, scale):
    # 提取可见关键点
    visible = true_keypoints[:, 2] != 0
    true_keypoints = true_keypoints[visible, :2]
    pred_keypoints = pred_keypoints[visible, :2]

    # 计算每个关键点的误差
    errors = np.linalg.norm(true_keypoints - pred_keypoints, axis=-1)

    # 计算平均误差
    mean_error = np.mean(errors)

    # 计算NME
    nme = mean_error / scale

    return nme


def compute_keypoint_nme(true_keypoints, pred_keypoints, scale):
    # 提取可见关键点
    visible = true_keypoints[:, 2] != 0
    true_keypoints = true_keypoints[:, :2]
    pred_keypoints = pred_keypoints[:, :2]

    # 计算每个关键点的误差
    errors = np.linalg.norm(true_keypoints - pred_keypoints, axis=-1)
    errors = np.array([vis * val for vis, val in zip(visible, errors)])

    # 计算NME
    nme = errors / scale

    return nme, visible


# 适用于从含有oks和pck的log文件中找到前20
def read_results(log_path,save_num=20, has_oks=True, has_pck=True):
    results = []
    with open(log_path, 'r') as file:
        for line in file:
            values = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
            if has_oks and has_pck:
                if len(values) >= 3:
                    epoch = int(values[0])
                    val_mean_oks = float(values[1])
                    val_mean_pck = float(values[2])
                    results.append((epoch, val_mean_oks, val_mean_pck))
            elif has_pck:
                epoch = int(values[0])
                # no oks
                val_mean_oks = epoch / 100
                val_mean_pck = float(values[1])
                results.append((epoch, val_mean_oks, val_mean_pck))
            elif has_oks:
                epoch = int(values[0])
                # no oks
                val_mean_oks = float(values[1])
                val_mean_pck = epoch / 100
                results.append((epoch, val_mean_oks, val_mean_pck))

    sorted_val_mean_oks = sorted(results, key=lambda x: x[1], reverse=True)
    sorted_val_mean_pck = sorted(results, key=lambda x: x[2], reverse=True)
    best_val_mean_oks_epochs = sorted_val_mean_oks[:save_num]
    best_val_mean_pck_epochs = sorted_val_mean_pck[:save_num]
    best_val_mean_oks_ids = [part[0] for part in best_val_mean_oks_epochs]
    best_val_mean_pck_ids = [part[0] for part in best_val_mean_pck_epochs]
    best_ids_set = set()
    for id_oks in best_val_mean_oks_ids:
        if id_oks not in best_ids_set:
            best_ids_set.add(id_oks)
    for id_pck in best_val_mean_pck_ids:
        if id_pck not in best_ids_set:
            best_ids_set.add(id_pck)
    return best_ids_set


def read_loss_results(log_path,num=20):
    results = []
    with open(log_path, 'r') as file:
        for line in file:
            values = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', line)
            if len(values) >= 3:
                epoch = int(values[0])
                train_mean_loss = float(values[1])
                eval_mean_loss = float(values[2])
                results.append((epoch, train_mean_loss, eval_mean_loss))

    sorted_eval_mean_loss = sorted(results, key=lambda x: x[2], reverse=False)
    best_eval_mean_loss = sorted_eval_mean_loss[:num]
    best_eval_mean_loss_ids = [part[0] for part in best_eval_mean_loss]
    best_ids_set = set()
    for id_pck in best_eval_mean_loss_ids:
        if id_pck not in best_ids_set:
            best_ids_set.add(id_pck)
    return best_ids_set


def weights_clean(model_base_path, ids_set):
    # 获取模型路径下的所有文件
    files = sorted(os.listdir(model_base_path))

    for file_name in files:
        if file_name.endswith(".pth"):  # 只处理以 .pth 结尾的文件
            # 提取文件名中的ID
            tmp_id = int(file_name.split("-")[1].split(".")[0])

            if tmp_id not in ids_set:  # 如果ID不在指定的ID集合中
                file_path = os.path.join(model_base_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)  # 删除文件
                    print(f"{file_name} has been deleted")
                else:
                    print(f"{file_name} doesn't exist")


def prediction_result_clean(results_base_path, ids_set):
    # 获取模型路径下的所有文件
    files = sorted(os.listdir(results_base_path))

    for file_name in files:
        if file_name.startswith("model-") and file_name.endswith("results.json"):  # 只处理以 .pth 结尾的文件
            # 提取文件名中的ID
            tmp_id = int(file_name.split("-")[1].split("_")[0])

            if tmp_id not in ids_set:  # 如果ID不在指定的ID集合中
                file_path = os.path.join(results_base_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)  # 删除文件
                    print(f"{file_name} has been deleted")
                else:
                    print(f"{file_name} doesn't exist")


# root_path = '../experiment/2023-08_08_20-46-37'
def models_clean(root_path, save_num=20,has_oks=True, has_pck=True):
    log_path = f'{root_path}/info/val_log.txt'
    model_base_path = f'{root_path}/save_weights'
    prediction_base_path = f'{root_path}/results'
    ids_set = read_results(log_path, save_num,has_oks, has_pck)
    weights_clean(model_base_path=model_base_path, ids_set=ids_set)
    prediction_result_clean(results_base_path=prediction_base_path, ids_set=ids_set)


def models_clean_by_loss(root_path):
    log_path = f'{root_path}/info/val_log.txt'
    model_base_path = f'{root_path}/save_weights'
    ids_set = read_loss_results(log_path)
    weights_clean(model_base_path=model_base_path, ids_set=ids_set)


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


# for ap_10k
# change the ratio of overlapping keypoints
def get_block_list(args,level):
    kps_a = ['R_eye', 'neck', 'L_F_hip', 'R_B_hip', 'R_F_knee', 'L_B_knee', 'L_F_paw', 'R_B_paw']
    kps_b = ['L_eye', 'nose', 'tail', 'R_F_hip', 'L_B_hip', 'L_F_knee', 'R_B_knee', 'R_F_paw', 'L_B_paw']
    common_kps_a = []
    common_kps_b = ['neck', 'L_F_hip', 'R_F_hip', 'L_B_hip', 'R_B_hip']
    common_kps_c = ['L_F_hip', 'R_F_hip', 'L_B_hip', 'R_B_hip', 'L_F_paw', 'R_F_paw', 'L_B_paw', 'R_B_paw']
    common_kps_d = ['L_F_hip', 'R_F_hip', 'L_B_hip', 'R_B_hip', 'L_F_paw', 'R_F_paw', 'L_B_paw', 'R_B_paw',
                    'nose', 'L_F_knee', 'R_F_knee', 'L_B_knee', 'R_B_knee']
    common_kps_e = ['L_eye', 'R_eye', 'nose', 'neck', 'tail', 'L_F_hip', 'L_F_knee', 'L_F_paw', 'R_F_hip',
                    'R_F_knee', 'R_F_paw', 'L_B_hip', 'L_B_knee', 'L_B_paw', 'R_B_hip', 'R_B_knee', 'R_B_paw']

    with open(args.keypoints_path, 'r') as f:
        animal_kps_info = json.load(f)
    kps_definition = animal_kps_info['keypoints']
    kps_a_index = []
    kps_b_index = []
    common_kps_a_index = []
    common_kps_b_index = []
    common_kps_c_index = []
    common_kps_d_index = []
    common_kps_e_index = []

    for kp_index, kp in enumerate(kps_definition):
        if kp in kps_a:
            kps_a_index.append(kp_index)
        if kp in kps_b:
            kps_b_index.append(kp_index)
        if kp in common_kps_a:
            common_kps_a_index.append(kp_index)
        if kp in common_kps_b:
            common_kps_b_index.append(kp_index)
        if kp in common_kps_c:
            common_kps_c_index.append(kp_index)
        if kp in common_kps_d:
            common_kps_d_index.append(kp_index)
        if kp in common_kps_e:
            common_kps_e_index.append(kp_index)

    common_kps_index = [common_kps_a_index, common_kps_b_index, common_kps_c_index, common_kps_d_index,
                        common_kps_e_index]

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


def get_block_list_v2(args,level):
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


def get_block_list_tmp(args,level):
    kps_a = ['R_eye', 'neck', 'L_F_hip', 'R_B_hip', 'R_F_knee', 'L_B_knee', 'L_F_paw', 'R_B_paw']
    kps_b = ['L_eye', 'nose', 'tail', 'R_F_hip', 'L_B_hip', 'L_F_knee', 'R_B_knee', 'R_F_paw', 'L_B_paw']
    common_kps_a = []
    common_kps_b = ['neck', 'L_F_hip', 'R_F_hip', 'L_B_hip', 'R_B_hip']
    common_kps_c = ['L_F_hip', 'R_F_hip', 'L_B_hip', 'R_B_hip', 'L_F_paw', 'R_F_paw', 'L_B_paw', 'R_B_paw']
    common_kps_d = ['L_F_hip', 'R_F_hip', 'L_B_hip', 'R_B_hip', 'L_F_paw', 'R_F_paw', 'L_B_paw', 'R_B_paw',
                    'nose', 'L_F_knee', 'R_F_knee', 'L_B_knee', 'R_B_knee']
    common_kps_e = ['L_eye', 'R_eye', 'nose', 'neck', 'tail', 'L_F_hip', 'L_F_knee', 'L_F_paw', 'R_F_hip',
                    'R_F_knee', 'R_F_paw', 'L_B_hip', 'L_B_knee', 'L_B_paw', 'R_B_hip', 'R_B_knee', 'R_B_paw']

    with open(args.keypoints_path, 'r') as f:
        animal_kps_info = json.load(f)
    kps_definition = animal_kps_info['keypoints']
    kps_a_index = []
    kps_b_index = []
    common_kps_a_index = []
    common_kps_b_index = []
    common_kps_c_index = []
    common_kps_d_index = []
    common_kps_e_index = []

    for kp_index, kp in enumerate(kps_definition):
        if kp in kps_a:
            kps_a_index.append(kp_index)
        if kp in kps_b:
            kps_b_index.append(kp_index)
        if kp in common_kps_a:
            common_kps_a_index.append(kp_index)
        if kp in common_kps_b:
            common_kps_b_index.append(kp_index)
        if kp in common_kps_c:
            common_kps_c_index.append(kp_index)
        if kp in common_kps_d:
            common_kps_d_index.append(kp_index)
        if kp in common_kps_e:
            common_kps_e_index.append(kp_index)

    common_kps_index = [common_kps_a_index, common_kps_b_index, common_kps_c_index, common_kps_d_index,
                        common_kps_e_index]

    common_kps_index_cur = common_kps_index[level]
    index_a = []
    index_b = []
    special_index_a = []
    special_index_b = []
    block_index_a = []
    block_index_b = []
    for i in range(17):
        if i in kps_a_index or i in common_kps_index_cur:
            index_a.append(i)
            if i not in common_kps_index_cur:
                special_index_a.append(i)
        if i in kps_b_index or i in common_kps_index_cur:
            index_b.append(i)
            if i not in common_kps_index_cur:
                special_index_b.append(i)
        if i not in kps_a_index and i not in common_kps_index_cur:
            block_index_a.append(i)
        if i not in kps_b_index and i not in common_kps_index_cur:
            block_index_b.append(i)

    return index_a,special_index_a,common_kps_index[level],index_b,special_index_b


# Loss_feedback = (s_loss_l_old - s_loss_l_new) * criterion(t_logits_us,hard_PL_us)
# 在mixing dataset中，由于点定义的问题，Loss_feedback更倾向于提高shared_kps的性能
# 此处将其拆解开，以AP-10K,Animal Pose的mixing dataset为例，此处拆成了shared_kps,ap_10k_exclusive_kps,animal_pose_exclusive_kps
def feedback_func(s_logits_old,s_logits_new,targets_heatmap,targets_visible,t_logits_us,hard_pl_us,hard_pl_visible,criterion,args):
    shared_kps_index = [0, 1, 4, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25]
    # shared_kps_index_a = [0, 1, 4]
    # shared_kps_index_b = [11, 12, 15, 16, 21, 22, 25]
    # shared_kps_index_c = [13, 14, 17, 18, 23, 24]
    ap_10k_exclusive_kps_index_a = [8]
    animal_pose_exclusive_kps_index_b = [2, 3, 6, 7]

    indices_list = [shared_kps_index,ap_10k_exclusive_kps_index_a,animal_pose_exclusive_kps_index_b]
    feedback_losses = []
    for indices in indices_list:
        s_loss_l_old = criterion(s_logits_old[:,indices].detach(),targets_heatmap[:,indices],targets_visible[:,indices],args,indices)
        s_loss_l_new = criterion(s_logits_new[:,indices].detach(),targets_heatmap[:,indices],targets_visible[:,indices],args,indices)
        dot_product = s_loss_l_old - s_loss_l_new
        feedback_loss = dot_product * criterion(t_logits_us[:,indices],hard_pl_us[:,indices],hard_pl_visible[:,indices],args,indices)
        feedback_losses.append(feedback_loss)
    return feedback_losses


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    # assert 0 <= current <= rampdown_length
    current = np.clip(current, 0.0, rampdown_length)
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def get_current_topkrate(epoch, rampdown_epoch, min_rate):
    r = cosine_rampdown(epoch, rampdown_epoch)
    return np.clip(r, min_rate, 1)


def calculate_weight(start_step, mid_step, end_step, weight, current_step):
    """
    :param start_step:
    :param mid_step: [start_step,end_step] 线性上升到weight
    :param end_step: [mid_step,end_step] 从weight余弦下降
    :param weight: 最大值
    :param current_step:
    :return:
    """
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


def generate_initial_pseudo_label(args,model,data_loader):
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)

    results = []
    with torch.no_grad():
        for batch in tqdm(data_loader,desc="Pseudo Label Generating"):
            imgs, targets = batch
            imgs = imgs.cuda()
            outputs = model(imgs)
            flipped_images = transforms.flip_images(imgs)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

            coords, maxvals = get_max_preds(outputs)
            maxvals = maxvals.float().squeeze(-1).cpu()
            maxvals = maxvals.numpy().tolist()
            coords = coords.cpu().numpy().tolist()

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)

            for output, target,coord,pl_confidence in zip(zip(outputs[0], outputs[1]), targets, coords, maxvals):
                result = convert_output_to_coco_format_batch_mix(output, target, num_joints=args.num_joints)
                result['coord'] = coord
                result['confidence'] = pl_confidence
                results.append(result)
    with open(args.pseudo_label_path,'w') as f:
        json.dump(results,f)


def get_topk_th_group(args,step,confidences,min_ratios):
    group_indice_face = [0, 1, 2]
    group_indice_body = [4, 5, 6, 7, 8, 9, 10,11, 12, 13, 14, 15, 16]
    group_indice_exclusive = [3, 17, 18, 19, 20]
    group_indices = [group_indice_face, group_indice_body, group_indice_exclusive]
    confidence_thresholds = []

    for i, indices in enumerate(group_indices):
        group_confidences = []
        for index in indices:
            for j in range(confidences.shape[0]):
                group_confidences.append(confidences[j][index].item())
        group_confidences = sorted(group_confidences, reverse=True)
        cur_ratio = get_current_topkrate(step, args.down_step, min_rate=min_ratios[i])
        sample_nums = len(group_confidences)
        cur_confidence_th = max(group_confidences[min(int(sample_nums * cur_ratio), sample_nums - 1)], 0.1)
        confidence_thresholds.append(cur_confidence_th)
    return confidence_thresholds,group_indices


def get_topk_th(args,step,confidences,min_ratios=0.85):
    group_confidences = []
    for i in range(confidences.shape[0]):
        for j in range(confidences.shape[1]):
            group_confidences.append(confidences[i][j].item())
    group_confidences = sorted(group_confidences, reverse=True)
    cur_ratio = get_current_topkrate(step, args.down_step, min_rate=min_ratios)
    sample_nums = len(group_confidences)
    cur_confidence_th = max(group_confidences[min(int(sample_nums * cur_ratio), sample_nums - 1)], 0.1)
    return cur_confidence_th


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description=__doc__)
    # parser.add_argument('--workers', default=4, type=int, help='number of workers for DataLoader')
    # parser.add_argument('--fixed-size', default=[256, 256], nargs='+', type=int, help='input size')
    # parser.add_argument('--val-data-path', default='../dataset/tigdog', type=str, help='data path')
    # parser.add_argument('--batch-size', default=32, type=int, help='train batch size')
    # parser.add_argument('--device', default='cuda:0', type=str, help='device info')
    # parser.add_argument('--output_dir', default='.', type=str, help='data path')
    # args = parser.parse_args()
    # # model_name = 'model-ap-10k-aug-best'
    # # model_name = 'model-animal-pose-best'
    # gt_keypoints_show_one_by_one(args, dataset='tigdog', mode='train')
    weights = 5
    start_step = 0
    end_step = 500
    mid_step = 50
    weights_ls = []
    for step in range(end_step):
        weights_ls.append(calculate_weight(start_step,mid_step,end_step,weights,step))
    plt.plot(weights_ls)
    plt.show()
