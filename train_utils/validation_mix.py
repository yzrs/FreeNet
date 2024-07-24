import argparse
import datetime
import logging
import math
import random
import time

import torch
import numpy as np
import json
import os
import sys

from matplotlib import pyplot as plt
from pycocotools.cocoeval import COCOeval
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_utils.augmentation import RandWeakAugment
from train_utils import transforms
from train_utils.utils import json_generate_batch, json_generate_batch_mix, compute_oks_mix, KpLossLabel, AverageMeter, \
    json_generate_batch_mix_key, json_generate_key_batch, \
    json_generate_batch_mix_key_parallel, json_generate_batch_mix_model_parallel
from train_utils.dataset import MixKeypoint


# dataset is a mix dataset with coco_list and valid_lists
def evaluate(args, model_name, dataset):
    with open(args.keypoints_path, 'r') as f:
        kps_definition = json.load(f)
    summary_output = f'{args.output_dir}/results/{model_name}_mix_dataset_performance.txt'

    anns_num_list = []
    oks_mean_list = []
    pck_mean_list = []
    for index, single_dataset in enumerate(dataset.dataset_infos):
        # OKS
        # 我需要判断dataset中有那些dataset作为验证集，并依次处理
        # 如果本地已经有预测结果  不再重复预测
        current_dataset = single_dataset['dataset']
        current_mode = single_dataset['mode']
        current_res_file = f'{args.output_dir}/results/{model_name}_mix_dataset_{current_dataset}_{current_mode}_results.json'

        # 根据数据集的长度取出子集对应的长度索引 构建子数据集
        start_index = 0
        for i in range(index):
            start_index += dataset.length_list[i]
        end_index = start_index + dataset.length_list[index]
        subset_indices = range(start_index, end_index)
        anns_num_list.append(dataset.length_list[index])
        # 创建子集数据对象 并生成对应的预测结果
        subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
        if not os.path.exists(current_res_file):
            json_generate_batch_mix(args, model_name, subset_dataset, current_res_file,num_joints=args.num_joints)

        # 用子coco进行预测
        coco_gt = dataset.coco_lists[index]['coco']
        coco_dt = coco_gt.loadRes(current_res_file)
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='keypoints')
        coco_eval.params.useSegm = None
        coco_eval.params.kpt_oks_sigmas = np.array(kps_definition[current_dataset]['sigmas'])
        coco_eval.evaluate()
        coco_eval.accumulate()
        # 保存OKS的评测结果
        with open(summary_output, 'a') as f:
            f.write("\nObject Keypoints Similarity Performance:\n")
            sys.stdout = f
            coco_eval.summarize()
            sys.stdout = sys.__stdout__

        # 保存OKS mAP value
        oks_mean_list.append(coco_eval.stats[0])

        # 计算对应的PCK@0.05
        pck_threshold = 0.05
        # image level pck
        pck_list = []
        for img_id in coco_gt.getImgIds():
            # get the image
            img = coco_gt.loadImgs(img_id)[0]

            # get the GT keypoints
            gt_ann_ids = coco_gt.getAnnIds(imgIds=img['id'])
            gt_anns = coco_gt.loadAnns(gt_ann_ids)

            # get the Pred keypoints
            dt_ann_ids = coco_dt.getAnnIds(imgIds=img['id'])
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            for gt_ann in gt_anns:
                for dt_ann in dt_anns:
                    if gt_ann['id'] == dt_ann['anno_id']:
                        if gt_ann['num_keypoints'] == 0:
                            continue
                        gt = gt_ann['keypoints']
                        dt = dt_ann['keypoints']
                        # reshape to (17,3)
                        gt = np.array(gt).reshape(-1, 3)
                        dt = np.array(dt).reshape(-1, 3)

                        scale = math.sqrt(gt_ann['area'])
                        dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                        # 忽略可见性为0的点
                        mask = gt[:, 2] > 0
                        pck_arr = dist[mask] < pck_threshold

                        # 计算布尔数组中True值的比例，小于阈值即为True
                        pck_list.append(np.sum(pck_arr) / len(pck_arr))
                        # print(np.sum(pck_arr),len(pck_arr))
                        break

        tmp_mean_pck = np.mean(pck_list)
        with open(summary_output, 'a') as f:
            f.write(f"Mean PCK on {current_dataset} {current_mode} Dataset:{tmp_mean_pck}")
        pck_mean_list.append(tmp_mean_pck)
        # print(f"{current_dataset} {current_mode} : {tmp_mean_pck}")

    with open(summary_output, 'r') as f:
        print(f.read())
    mean_oks = np.average(oks_mean_list, weights=anns_num_list)
    mean_pck = np.average(pck_mean_list, weights=anns_num_list)
    return mean_oks, mean_pck, oks_mean_list, pck_mean_list


def eval_key(args, model_name, dataset,key):
    with open(args.keypoints_path, 'r') as f:
        kps_definition = json.load(f)
    summary_output = f'{args.output_dir}/results/{model_name}_{key}_mix_dataset_performance.txt'

    anns_num_list = []
    oks_mean_list = []
    pck_mean_list = []
    for index, single_dataset in enumerate(dataset.dataset_infos):
        # OKS
        # 我需要判断dataset中有那些dataset作为验证集，并依次处理
        # 如果本地已经有预测结果  不再重复预测
        current_dataset = single_dataset['dataset']
        current_mode = single_dataset['mode']
        current_res_file = f'{args.output_dir}/results/{model_name}_{key}_mix_dataset_{current_dataset}_{current_mode}_results.json'

        # 根据数据集的长度取出子集对应的长度索引 构建子数据集
        start_index = 0
        for i in range(index):
            start_index += dataset.length_list[i]
        end_index = start_index + dataset.length_list[index]
        subset_indices = range(start_index, end_index)
        anns_num_list.append(dataset.length_list[index])
        # 创建子集数据对象 并生成对应的预测结果
        subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
        if not os.path.exists(current_res_file):
            json_generate_batch_mix_key(args, model_name, subset_dataset, current_res_file,key=key,num_joints=args.num_joints)

        # 用子coco进行预测
        coco_gt = dataset.coco_lists[index]['coco']
        coco_dt = coco_gt.loadRes(current_res_file)
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='keypoints')
        coco_eval.params.useSegm = None
        coco_eval.params.kpt_oks_sigmas = np.array(kps_definition[current_dataset]['sigmas'])
        coco_eval.evaluate()
        coco_eval.accumulate()
        # 保存OKS的评测结果
        with open(summary_output, 'a') as f:
            f.write("\nObject Keypoints Similarity Performance:\n")
            sys.stdout = f
            coco_eval.summarize()
            sys.stdout = sys.__stdout__

        # 保存OKS mAP value
        oks_mean_list.append(coco_eval.stats[0])

        # 计算对应的PCK@0.05
        pck_threshold = 0.05
        # image level pck
        pck_list = []
        for img_id in coco_gt.getImgIds():
            # get the image
            img = coco_gt.loadImgs(img_id)[0]

            # get the GT keypoints
            gt_ann_ids = coco_gt.getAnnIds(imgIds=img['id'])
            gt_anns = coco_gt.loadAnns(gt_ann_ids)

            # get the Pred keypoints
            dt_ann_ids = coco_dt.getAnnIds(imgIds=img['id'])
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            for gt_ann in gt_anns:
                for dt_ann in dt_anns:
                    if gt_ann['id'] == dt_ann['anno_id']:
                        if gt_ann['num_keypoints'] == 0:
                            continue
                        gt = gt_ann['keypoints']
                        dt = dt_ann['keypoints']
                        # reshape to (17,3)
                        gt = np.array(gt).reshape(-1, 3)
                        dt = np.array(dt).reshape(-1, 3)

                        scale = math.sqrt(gt_ann['area'])
                        dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                        # 忽略可见性为0的点
                        mask = gt[:, 2] > 0
                        pck_arr = dist[mask] < pck_threshold

                        # 计算布尔数组中True值的比例，小于阈值即为True
                        pck_list.append(np.sum(pck_arr) / len(pck_arr))
                        # print(np.sum(pck_arr),len(pck_arr))
                        break

        tmp_mean_pck = np.mean(pck_list)
        with open(summary_output, 'a') as f:
            f.write(f"Mean PCK on {current_dataset} {current_mode} Dataset:{tmp_mean_pck}")
        pck_mean_list.append(tmp_mean_pck)
        # print(f"{current_dataset} {current_mode} : {tmp_mean_pck}")

    with open(summary_output, 'r') as f:
        print(f.read())
    mean_oks = np.average(oks_mean_list, weights=anns_num_list)
    mean_pck = np.average(pck_mean_list, weights=anns_num_list)
    return mean_oks, mean_pck, oks_mean_list, pck_mean_list


def eval_key_parallel(args, model_name, dataset,key):
    with open(args.keypoints_path, 'r') as f:
        kps_definition = json.load(f)
    summary_output = f'{args.output_dir}/results/{model_name}_{key}_mix_dataset_performance.txt'

    anns_num_list = []
    oks_mean_list = []
    pck_mean_list = []
    for index, single_dataset in enumerate(dataset.dataset_infos):
        # OKS
        # 我需要判断dataset中有那些dataset作为验证集，并依次处理
        # 如果本地已经有预测结果  不再重复预测
        current_dataset = single_dataset['dataset']
        current_mode = single_dataset['mode']
        current_res_file = f'{args.output_dir}/results/{model_name}_{key}_mix_dataset_{current_dataset}_{current_mode}_results.json'

        # 根据数据集的长度取出子集对应的长度索引 构建子数据集
        start_index = 0
        for i in range(index):
            start_index += dataset.length_list[i]
        end_index = start_index + dataset.length_list[index]
        subset_indices = range(start_index, end_index)
        anns_num_list.append(dataset.length_list[index])
        # 创建子集数据对象 并生成对应的预测结果
        subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
        if not os.path.exists(current_res_file):
            json_generate_batch_mix_key_parallel(args, model_name, subset_dataset, current_res_file,key=key)

        # 用子coco进行预测
        coco_gt = dataset.coco_lists[index]['coco']
        coco_dt = coco_gt.loadRes(current_res_file)
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='keypoints')
        coco_eval.params.useSegm = None
        coco_eval.params.kpt_oks_sigmas = np.array(kps_definition[current_dataset]['sigmas'])
        coco_eval.evaluate()
        coco_eval.accumulate()
        # 保存OKS的评测结果
        with open(summary_output, 'a') as f:
            f.write("\nObject Keypoints Similarity Performance:\n")
            sys.stdout = f
            coco_eval.summarize()
            sys.stdout = sys.__stdout__

        # 保存OKS mAP value
        oks_mean_list.append(coco_eval.stats[0])

        # 计算对应的PCK@0.05
        pck_threshold = 0.05
        # image level pck
        pck_list = []
        for img_id in coco_gt.getImgIds():
            # get the image
            img = coco_gt.loadImgs(img_id)[0]

            # get the GT keypoints
            gt_ann_ids = coco_gt.getAnnIds(imgIds=img['id'])
            gt_anns = coco_gt.loadAnns(gt_ann_ids)

            # get the Pred keypoints
            dt_ann_ids = coco_dt.getAnnIds(imgIds=img['id'])
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            for gt_ann in gt_anns:
                for dt_ann in dt_anns:
                    if gt_ann['id'] == dt_ann['anno_id']:
                        if gt_ann['num_keypoints'] == 0:
                            continue
                        gt = gt_ann['keypoints']
                        dt = dt_ann['keypoints']
                        # reshape to (17,3)
                        gt = np.array(gt).reshape(-1, 3)
                        dt = np.array(dt).reshape(-1, 3)

                        scale = math.sqrt(gt_ann['area'])
                        dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                        # 忽略可见性为0的点
                        mask = gt[:, 2] > 0
                        pck_arr = dist[mask] < pck_threshold

                        # 计算布尔数组中True值的比例，小于阈值即为True
                        pck_list.append(np.sum(pck_arr) / len(pck_arr))
                        # print(np.sum(pck_arr),len(pck_arr))
                        break

        tmp_mean_pck = np.mean(pck_list)
        with open(summary_output, 'a') as f:
            f.write(f"Mean PCK on {current_dataset} {current_mode} Dataset:{tmp_mean_pck}")
        pck_mean_list.append(tmp_mean_pck)
        # print(f"{current_dataset} {current_mode} : {tmp_mean_pck}")

    with open(summary_output, 'r') as f:
        print(f.read())
    mean_oks = np.average(oks_mean_list, weights=anns_num_list)
    mean_pck = np.average(pck_mean_list, weights=anns_num_list)
    return mean_oks, mean_pck, oks_mean_list, pck_mean_list


def eval_model_parallel(args, model,model_name, dataset,key):
    with open(args.keypoints_path, 'r') as f:
        kps_definition = json.load(f)
    summary_output = f'{args.output_dir}/results/{model_name}_{key}_mix_dataset_performance.txt'

    anns_num_list = []
    oks_mean_list = []
    pck_mean_list = []
    for index, single_dataset in enumerate(dataset.dataset_infos):
        # OKS
        # 我需要判断dataset中有那些dataset作为验证集，并依次处理
        # 如果本地已经有预测结果  不再重复预测
        current_dataset = single_dataset['dataset']
        current_mode = single_dataset['mode']
        current_res_file = f'{args.output_dir}/results/{model_name}_{key}_mix_dataset_{current_dataset}_{current_mode}_results.json'

        # 根据数据集的长度取出子集对应的长度索引 构建子数据集
        start_index = 0
        for i in range(index):
            start_index += dataset.length_list[i]
        end_index = start_index + dataset.length_list[index]
        subset_indices = range(start_index, end_index)
        anns_num_list.append(dataset.length_list[index])
        # 创建子集数据对象 并生成对应的预测结果
        subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
        if not os.path.exists(current_res_file):
            json_generate_batch_mix_model_parallel(args, model, subset_dataset,current_res_file,num_joints=args.num_joints)

        # 用子coco进行预测
        coco_gt = dataset.coco_lists[index]['coco']
        coco_dt = coco_gt.loadRes(current_res_file)
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='keypoints')
        coco_eval.params.useSegm = None
        coco_eval.params.kpt_oks_sigmas = np.array(kps_definition[current_dataset]['sigmas'])
        coco_eval.evaluate()
        coco_eval.accumulate()
        # 保存OKS的评测结果
        with open(summary_output, 'a') as f:
            f.write("\nObject Keypoints Similarity Performance:\n")
            sys.stdout = f
            coco_eval.summarize()
            sys.stdout = sys.__stdout__

        # 保存OKS mAP value
        oks_mean_list.append(coco_eval.stats[0])

        # 计算对应的PCK@0.05
        pck_threshold = 0.05
        # image level pck
        pck_list = []
        for img_id in coco_gt.getImgIds():
            # get the image
            img = coco_gt.loadImgs(img_id)[0]

            # get the GT keypoints
            gt_ann_ids = coco_gt.getAnnIds(imgIds=img['id'])
            gt_anns = coco_gt.loadAnns(gt_ann_ids)

            # get the Pred keypoints
            dt_ann_ids = coco_dt.getAnnIds(imgIds=img['id'])
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            for gt_ann in gt_anns:
                for dt_ann in dt_anns:
                    if gt_ann['id'] == dt_ann['anno_id']:
                        if gt_ann['num_keypoints'] == 0:
                            continue
                        gt = gt_ann['keypoints']
                        dt = dt_ann['keypoints']
                        # reshape to (17,3)
                        gt = np.array(gt).reshape(-1, 3)
                        dt = np.array(dt).reshape(-1, 3)

                        scale = math.sqrt(gt_ann['area'])
                        dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                        # 忽略可见性为0的点
                        mask = gt[:, 2] > 0
                        pck_arr = dist[mask] < pck_threshold

                        # 计算布尔数组中True值的比例，小于阈值即为True
                        pck_list.append(np.sum(pck_arr) / len(pck_arr))
                        # print(np.sum(pck_arr),len(pck_arr))
                        break

        tmp_mean_pck = np.mean(pck_list)
        with open(summary_output, 'a') as f:
            f.write(f"Mean PCK on {current_dataset} {current_mode} Dataset:{tmp_mean_pck}")
        pck_mean_list.append(tmp_mean_pck)

    with open(summary_output, 'r') as f:
        print(f.read())
    mean_oks = np.average(oks_mean_list, weights=anns_num_list)
    mean_pck = np.average(pck_mean_list, weights=anns_num_list)
    return mean_oks, mean_pck, oks_mean_list, pck_mean_list


def evaluate_pck(args, model_name, dataset):
    with open(args.keypoints_path, 'r') as f:
        kps_definition = json.load(f)
    summary_output = f'{args.output_dir}/results/{model_name}_mix_dataset_performance.txt'

    anns_num_list = []
    pck_mean_list = []
    for index, single_dataset in enumerate(dataset.dataset_infos):
        # 我需要判断dataset中有那些dataset作为验证集，并依次处理
        # 如果本地已经有预测结果  不再重复预测
        current_dataset = single_dataset['dataset']
        current_mode = single_dataset['mode']
        current_res_file = f'{args.output_dir}/results/{model_name}_mix_dataset_{current_dataset}_{current_mode}_results.json'

        # 根据数据集的长度取出子集对应的长度索引 构建子数据集
        start_index = 0
        for i in range(index):
            start_index += dataset.length_list[i]
        end_index = start_index + dataset.length_list[index]
        subset_indices = range(start_index, end_index)
        anns_num_list.append(dataset.length_list[index])
        # 创建子集数据对象 并生成对应的预测结果
        subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
        if not os.path.exists(current_res_file):
            json_generate_batch_mix(args, model_name, subset_dataset, current_res_file)

        # 用子coco进行预测
        coco_gt = dataset.coco_lists[index]['coco']
        coco_dt = coco_gt.loadRes(current_res_file)

        # 计算对应的PCK@0.05
        pck_threshold = 0.05
        # image level pck
        pck_list = []

        for img_id in coco_gt.getImgIds():
            # get the image
            img = coco_gt.loadImgs(img_id)[0]

            # get the GT keypoints
            gt_ann_ids = coco_gt.getAnnIds(imgIds=img['id'])
            gt_anns = coco_gt.loadAnns(gt_ann_ids)

            # get the Pred keypoints
            dt_ann_ids = coco_dt.getAnnIds(imgIds=img['id'])
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            # 如果没有对应的 dt 注释，跳过处理
            if len(dt_ann_ids) == 0:
                continue

            for gt_ann in gt_anns:
                for dt_ann in dt_anns:
                    if gt_ann['id'] == dt_ann['anno_id']:
                        if gt_ann['num_keypoints'] == 0:
                            continue

                        gt = gt_ann['keypoints']
                        dt = dt_ann['keypoints']
                        # reshape to (17,3)
                        gt = np.array(gt).reshape(-1, 3)
                        dt = np.array(dt).reshape(-1, 3)

                        scale = math.sqrt(gt_ann['area'])
                        dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                        # 忽略可见性为0的点
                        mask = gt[:, 2] > 0
                        pck_arr = dist[mask] < pck_threshold

                        # 计算布尔数组中True值的比例，小于阈值即为True
                        pck_list.append(np.sum(pck_arr) / len(pck_arr))
                        # print(np.sum(pck_arr),len(pck_arr))
                        break

        tmp_mean_pck = np.mean(pck_list)
        with open(summary_output, 'a') as f:
            f.write(f"Mean PCK on {current_dataset} {current_mode} Dataset:{tmp_mean_pck}.")
        pck_mean_list.append(tmp_mean_pck)
        # print(f"{current_dataset} {current_mode} : {tmp_mean_pck}")

    with open(summary_output, 'r') as f:
        print(f.read())
    mean_pck = np.average(pck_mean_list, weights=anns_num_list)
    return mean_pck, pck_mean_list


def evaluate_animal(args, model_name, dataset):
    out_path = f'{args.output_dir}/statistics_file/{model_name}_mix_dataset_animals_results.json'
    pck_mean_list = []
    for index, single_dataset in enumerate(dataset.dataset_infos):
        # 我需要判断dataset中有那些dataset作为验证集，并依次处理
        # 如果本地已经有预测结果  不再重复预测
        current_dataset = single_dataset['dataset']
        current_mode = single_dataset['mode']
        current_res_file = f'{args.output_dir}/results/{model_name}_mix_dataset_{current_dataset}_{current_mode}_results.json'

        # 根据数据集的长度取出子集对应的长度索引 构建子数据集
        start_index = 0
        for i in range(index):
            start_index += dataset.length_list[i]
        end_index = start_index + dataset.length_list[index]
        subset_indices = range(start_index, end_index)
        # 创建子集数据对象 并生成对应的预测结果
        subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
        if not os.path.exists(current_res_file):
            json_generate_batch_mix(args, model_name, subset_dataset, current_res_file)

        # 用子coco进行预测
        coco_gt = dataset.coco_lists[index]['coco']
        coco_dt = coco_gt.loadRes(current_res_file)

        # 计算对应的PCK@0.05
        pck_threshold = 0.05
        # image level pck
        pck_list = [[] for _ in range(len(coco_gt.cats))]

        for img_id in coco_gt.getImgIds():
            # get the image
            img = coco_gt.loadImgs(img_id)[0]

            # get the GT keypoints
            gt_ann_ids = coco_gt.getAnnIds(imgIds=img['id'])
            gt_anns = coco_gt.loadAnns(gt_ann_ids)

            # get the Pred keypoints
            dt_ann_ids = coco_dt.getAnnIds(imgIds=img['id'])
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            # 如果没有对应的 dt 注释，跳过处理
            if len(dt_ann_ids) == 0:
                continue

            for gt_ann in gt_anns:
                for dt_ann in dt_anns:
                    if gt_ann['id'] == dt_ann['anno_id']:
                        if gt_ann['num_keypoints'] == 0:
                            continue
                        category = gt_ann['category_id']

                        gt = gt_ann['keypoints']
                        dt = dt_ann['keypoints']
                        # reshape to (17,3)
                        gt = np.array(gt).reshape(-1, 3)
                        dt = np.array(dt).reshape(-1, 3)

                        scale = math.sqrt(gt_ann['area'])
                        dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                        # 忽略可见性为0的点
                        mask = gt[:, 2] > 0
                        pck_arr = dist[mask] < pck_threshold

                        # 计算布尔数组中True值的比例，小于阈值即为True
                        pck_list[category-1].append(np.sum(pck_arr) / len(pck_arr))
                        # print(np.sum(pck_arr),len(pck_arr))
                        break
        animals_info = coco_gt.cats
        animals = [animals_info[key]['name'] for key in animals_info]
        tmp_mean_pck = [np.mean(sublist) if len(sublist) > 0 else 0.0 for sublist in pck_list]
        animals_pck = {animal:pck for animal,pck in zip(animals,tmp_mean_pck)}
        pck_mean_list.append(animals_pck)
    with open(out_path,'w') as f:
        json.dump(pck_mean_list,f,indent=4)


# 返回kp_index对应的一组关键点在eval_index上的数据集的性能，若eval_index无效，则返回多个数据集上的均值
def eval_group_pck(args,model_name,val_dataset,kp_index,eval_index,key=None,threshold=0.05):
    if len(kp_index) == 0 or len(eval_index) == 0:
        return 0

    with open(args.keypoints_path,'r') as f:
        definition_info = json.load(f)

    # 保存所有目标的pck
    pck_group_list = [[[] for _ in range(len(kp_index))] for _ in range(len(val_dataset.dataset_infos))]

    for index, single_dataset in enumerate(val_dataset.dataset_infos):
        # 我需要判断dataset中有那些dataset作为验证集，并依次处理
        current_dataset = single_dataset['dataset']
        current_mode = single_dataset['mode']
        current_num_joints = single_dataset['num_joints']
        if key is None:
            current_res_file = f'{args.output_dir}/results/{model_name}_mix_dataset_{current_dataset}_{current_mode}_results.json'
            mix_current_res_file = f'{args.output_dir}/results/{model_name}_mix_dataset_{current_dataset}_{current_mode}_mix_results.json'
        else:
            current_res_file = f'{args.output_dir}/results/{model_name}_{key}_mix_dataset_{current_dataset}_{current_mode}_results.json'
            mix_current_res_file = f'{args.output_dir}/results/{model_name}_{key}_mix_dataset_{current_dataset}_{current_mode}_mix_results.json'

        map_pair = definition_info[current_dataset]['map']

        # group
        # 根据数据集的长度取出子集对应的长度索引 构建子数据集
        start_index = 0
        for i in range(index):
            start_index += val_dataset.length_list[i]
        end_index = start_index + val_dataset.length_list[index]
        subset_indices = range(start_index, end_index)
        # 创建子集数据对象 并生成对应的预测结果
        subset_dataset = torch.utils.data.Subset(val_dataset, subset_indices)
        if not os.path.exists(current_res_file):
            if key is None:
                json_generate_batch_mix(args, model_name, subset_dataset, current_res_file)
            else:
                json_generate_batch_mix_key(args,model_name,subset_dataset,current_res_file,key)
        if args.all_results:
            if not os.path.exists(mix_current_res_file):
                if key is None:
                    json_generate_batch(args, model_name, subset_dataset, mix_current_res_file)
                else:
                    json_generate_key_batch(args, model_name, subset_dataset,key,mix_current_res_file)

        # 用子coco进行预测
        coco_gt = val_dataset.coco_lists[index]['coco']
        coco_dt = coco_gt.loadRes(current_res_file)
        pck_threshold = threshold
        # 记录每个visible keypoint 的 true or false
        pck_list = [[] for _ in range(current_num_joints)]

        for img_id in coco_gt.getImgIds():
            # get the image
            img = coco_gt.loadImgs(img_id)[0]

            # get the GT keypoints
            gt_ann_ids = coco_gt.getAnnIds(imgIds=img['id'], iscrowd=None)
            gt_anns = coco_gt.loadAnns(gt_ann_ids)

            # get the Pred keypoints
            dt_ann_ids = coco_dt.getAnnIds(imgIds=img['id'], iscrowd=None)
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            for gt_ann in gt_anns:
                for dt_ann in dt_anns:
                    if gt_ann['id'] == dt_ann['anno_id']:
                        gt = gt_ann['keypoints']
                        dt = dt_ann['keypoints']
                        # reshape to (17,3)
                        gt = np.array(gt).reshape(current_num_joints, 3)
                        dt = np.array(dt).reshape(current_num_joints, 3)
                        # compute the normal 欧氏 distance
                        # 这里我使用两眼的间距作归一化参考
                        # 如果缺失眼部数据则采用平均眼距
                        scale = math.sqrt(gt_ann['area'])
                        dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                        # 忽略可见性为0的点
                        vis = gt[:, 2]
                        for i in range(current_num_joints):
                            if vis[i] != 0:
                                if dist[i] < pck_threshold:
                                    pck_list[i].append(1)
                                else:
                                    pck_list[i].append(0)

        # 得到每个点的pck后，进行分组的计算
        # 如果不是目标的测试集，则跳过
        # kp_index based on 26 keypoints definition
        # pck list based on 17 , 20 keypoints definition
        # A transform process is needed.
        if index in eval_index:
            for ind,mix_index in enumerate(kp_index):
                kp_ind = -1
                found = False
                for pair in map_pair:
                    if pair[1] == mix_index:
                        kp_ind = pair[0]
                        found = True
                        break
                if found:
                    pck_group_list[index][ind].extend(pck_list[kp_ind])

    # 计算每个点以及平均的PCK 在每个数据集上

    # 计算所有预测结果的整体均值
    flatten_all_predictions = [pck for sublist in pck_group_list for item in sublist for pck in item]  # 将所有预测结果放入一个大列表
    dataset_means = []
    for dataset_index in eval_index:
        dataset = pck_group_list[dataset_index]
        dataset_mean = []
        for point in dataset:
            dataset_mean.extend(point)
        if len(dataset_mean) > 0:
            dataset_means.append(np.mean(dataset_mean))

    if len(flatten_all_predictions) > 0:
        avg_pck = np.mean(flatten_all_predictions)
    else:
        avg_pck = 0
    print("AVG PCK@0.05 of this group:",avg_pck)
    print("AVG PCK@0.05 of every dataset:",dataset_means)
    return avg_pck,dataset_means


def test(args):
    data_root = args.data_root
    logger.info(dict(args._get_kwargs()))
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    fixed_size = args.fixed_size
    heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)
    kps_weights = np.array(animal_kps_info["kps_weights"],
                           dtype=np.float32).reshape((args.num_joints,))
    data_transform = {
        "train": transforms.Compose([
            transforms.LabelFormatTrans(extend_flag=True),
            transforms.HalfBody(0.3, animal_kps_info["upper_body_ids"], animal_kps_info["lower_body_ids"]),
            transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            transforms.RandomHorizontalFlip(0.5, animal_kps_info["flip_pairs"]),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            RandWeakAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.LabelFormatTrans(extend_flag=True),
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    val_dataset_info = [{"dataset":"ap_10k","mode":"val"},{"dataset":"animal_pose","mode":"val"}]
    # val_dataset_info = [{"dataset":"tigdog_horse","mode":"val"},{"dataset":"tigdog_tiger","mode":"val"}]
    # val_dataset_info = [{"dataset":"ap_10k","mode":"val"},{"dataset":"tigdog_horse","mode":"val"}]
    # val_dataset_info = [{"dataset":"ap_10k","mode":"test"}]
    val_dataset = MixKeypoint(root=data_root,merge_info=val_dataset_info,transform=data_transform['val'])
    name = '29K_mix_SL_best'
    # evaluate_keypoint_all(args,name,val_dataset)
    # evaluate_animal(args,name,val_dataset)
    # evaluate_keypoint(args,name,val_dataset)
    # evaluate_group(args,name,val_dataset)
    # evaluate(args,name,val_dataset)
    shared_kp_index = [0, 1, 4, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25]
    exclusive_kp_index_a = [8]
    exclusive_kp_index_b = [2, 3, 6, 7]
    eval_group_pck(args,name,val_dataset,shared_kp_index,eval_index=[0,1])
    eval_group_pck(args,name,val_dataset,exclusive_kp_index_a,eval_index=[0])
    eval_group_pck(args,name,val_dataset,exclusive_kp_index_b,eval_index=[1])


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--name', default='HRNet_Supervise_Mix_Keypoints', type=str, help='experiment name')
    parser.add_argument('--info', default="full", type=str, help='experiment info')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--data-root', default='../dataset', type=str, help='data path')
    parser.add_argument('--pretrained-model-path', default='./pretrained_weights',
                        type=str, help='pretrained weights path')
    parser.add_argument('--output-dir', default='.',type=str, help='output dir depends on the time')
    # parser.add_argument('--resume', default='./save_weights/model-209.pth', type=str, help='path to resume file')
    #
    parser.add_argument('--workers', default=2, type=int, help='number of workers for DataLoader')
    parser.add_argument('--batch-size', default=32, type=int, help='train batch size')

    parser.add_argument('--seed', default=2, type=int, help='seed for initializing training')
    # animal body关键点信息
    parser.add_argument('--keypoints-path', default="./info/keypoints_definition.json", type=str,
                        help='keypoints_format.json path')
    parser.add_argument('--fixed-size', default=[256, 256], nargs='+', type=int, help='input size')
    # keypoints点数
    parser.add_argument('--num-joints', default=26, type=int, help='num_joints')
    parser.add_argument("--all_results", default=False,action="store_true", help="if true, evaluation will also save 26 kps prediction")
    parser.add_argument("--amp", default=True,action="store_true", help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    now = datetime.datetime.now()
    output_dir = '..'

    info_output_dir = os.path.join(output_dir, '../info')
    if not os.path.exists(info_output_dir):
        os.mkdir(info_output_dir)
    results_output_dir = os.path.join(output_dir, '../results')
    if not os.path.exists(results_output_dir):
        os.mkdir(results_output_dir)
    save_weights_output_dir = os.path.join(output_dir, '../save_weights')
    if not os.path.exists(save_weights_output_dir):
        os.mkdir(save_weights_output_dir)

    print(args)
    if args.workers > 0:
        args.workers = min([os.cpu_count(), args.workers])
    else:
        args.workers = os.cpu_count()
    test(args)
