import argparse
import math
import torch
import numpy as np
import json
import os
import sys
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from train_utils.dataset import CocoKeypoint
from train_utils import transforms
from train_utils.utils import (compute_oks, compute_nme, compute_keypoint_nme, json_generate_batch,json_generate_key_batch,
                               compute_oks_no_threshold, KpLossLabel, AverageMeter,json_generate)


def eval_model_oks(args,model,name,dataset="ap_10k",mode="val"):
    # 加载 COCO ground truth 数据
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
                transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode,transform=data_transform["test"],
                          fixed_size=args.fixed_size,data_type="keypoints",num_joints=args.num_joints)
    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{name}_{dataset}_{mode}_results.json'

    if not os.path.exists(res_file):
        json_generate(args,model,cocoGt,res_file)

    cocoDt = cocoGt.coco.loadRes(res_file)
    coco_eval = COCOeval(cocoGt=cocoGt.coco,cocoDt=cocoDt,iouType='keypoints')

    coco_eval.params.useSegm = None
    with open(args.keypoints_path, 'r') as f:
        kps_definition = json.load(f)
    coco_eval.params.kpt_oks_sigmas = np.array(kps_definition['sigmas'])
    coco_eval.evaluate()
    coco_eval.accumulate()

    summary_output = f'{args.output_dir}/results/{name}_{dataset}_{mode}_performance.txt'
    with open(summary_output,'a') as f:
        f.write("Object Keypoints Similarity Performance:\n")
        sys.stdout = f
        coco_eval.summarize()
        sys.stdout = sys.__stdout__
    return coco_eval.stats[0]


def eval_model_pck(args,model,name,dataset="ap_10k",mode="val",threshold=0.05):
    # 加载 COCO ground truth 数据
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode,transform=data_transform["test"],
                          fixed_size=args.fixed_size,data_type="keypoints",num_joints=args.num_joints)
    # test_dataset_loader = DataLoader(cocoGt,
    #                                  batch_size=1,
    #                                  shuffle=False,
    #                                  pin_memory=True,
    #                                  num_workers=8,
    #                                  collate_fn=cocoGt.collate_fn)

    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{name}_{dataset}_{mode}_results.json'

    if not os.path.exists(res_file):
        json_generate(args,model,cocoGt,res_file)

    cocoDt = cocoGt.coco.loadRes(res_file)
    pck_threshold = threshold

    # image level pck
    pck_list = []
    for img_id in cocoGt.coco.getImgIds():
        # get the image
        img = cocoGt.coco.loadImgs(img_id)[0]

        # get the GT keypoints
        gt_ann_ids = cocoGt.coco.getAnnIds(imgIds=img['id'])
        gt_anns = cocoGt.coco.loadAnns(gt_ann_ids)

        # get the Pred keypoints
        dt_ann_ids = cocoDt.getAnnIds(imgIds=img['id'])
        dt_anns = cocoDt.loadAnns(dt_ann_ids)

        for gt_ann in gt_anns:
            for dt_ann in dt_anns:
                if gt_ann['id'] == dt_ann['anno_id']:
                    gt = gt_ann['keypoints']
                    dt = dt_ann['keypoints']
                    # reshape to (17,3)
                    gt = np.array(gt).reshape(args.num_joints,3)
                    dt = np.array(dt).reshape(args.num_joints,3)
                    # compute the normal 欧氏 distance
                    # 这里我使用两眼的间距作归一化参考
                    # 如果缺失眼部数据则采用平均眼距
                    # if gt[0,2] == 0 or gt[1,2] == 0:
                    #     eye_dist = avg_norm_dist
                    # else:
                    #     eye_dist = np.linalg.norm(gt[0,:2]-gt[1,:2])
                    # dist = np.linalg.norm(gt[:,:2]-dt[:,:2],axis=-1) / eye_dist
                    scale = math.sqrt(gt_ann['area'])
                    dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                    # 忽略可见性为0的点
                    mask = gt[:,2] > 0
                    # 计算布尔数组中True值的比例，小于阈值即为True
                    pck_arr = dist[mask] < pck_threshold
                    pck_list.append(np.sum(pck_arr) / len(pck_arr))

    # compute average pck value
    mean_pck = np.mean(pck_list)
    output_path = f'{args.output_dir}/results/{name}_{dataset}_{mode}_performance.txt'
    with open(output_path,'a') as f:
        f.write(f"\nMean PCK on Test Dataset:{mean_pck}\n")
    print(f"Mean PCK on Test Dataset:{mean_pck}")
    return mean_pck


# 在animal pose dataset的label data上计算模型的Object Keypoints Similarity指标
def oks_performance_test(args,model_name,dataset="ap_10k",mode="val"):
    # 加载 COCO ground truth 数据
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
                transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode,transform=data_transform["test"],
                          fixed_size=args.fixed_size,data_type="keypoints",num_joints=args.num_joints)
    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_results.json'

    if not os.path.exists(res_file):
        json_generate_batch(args,model_name,cocoGt,res_file)

    cocoDt = cocoGt.coco.loadRes(res_file)
    coco_eval = COCOeval(cocoGt=cocoGt.coco,cocoDt=cocoDt,iouType='keypoints')

    coco_eval.params.useSegm = None
    with open(args.keypoints_path, 'r') as f:
        kps_definition = json.load(f)
    # print(coco_eval.params.kpt_oks_sigmas)
    coco_eval.params.kpt_oks_sigmas = np.array(kps_definition['sigmas'])
    # print(coco_eval.params.kpt_oks_sigmas)
    # print(dir(coco_eval.params))

    coco_eval.evaluate()
    coco_eval.accumulate()

    summary_output = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_performance.txt'
    with open(summary_output,'a') as f:
        f.write("Object Keypoints Similarity Performance:\n")
        sys.stdout = f
        coco_eval.summarize()
        sys.stdout = sys.__stdout__
    return coco_eval.stats[0]


def oks_performance_test_keypoint(args,model_name,dataset="ap_10k",mode="val"):
    # 加载 COCO ground truth 数据
    with open(args.keypoints_path,'r') as f:
        dataset_definition = json.load(f)
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
                transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode,transform=data_transform["test"],
                          fixed_size=args.fixed_size,data_type="keypoints",num_joints=args.num_joints)
    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_results.json'

    if not os.path.exists(res_file):
        json_generate_batch(args,model_name,cocoGt,res_file)

    if dataset == 'ap_10k':
        keypoints = ["left_eye", "right_eye", "nose", "neck", "root_of_tail", "left_shoulder", "left_elbow", "L_F_Paw",
                     "right_shoulder", "right_elbow", "R_F_Paw", "left_hip", "left_knee", "L_B_Paw", "right_hip",
                     "right_knee", "R_B_Paw"]
        group = {"eye":{"left_eye","right_eye"},
                 "paw":{"L_F_Paw","L_B_Paw","R_F_Paw","R_B_Paw"},
                 "knee":{"left_elbow", "right_elbow", "left_knee","right_knee"},
                 "hip":{"left_shoulder","right_shoulder","left_hip","right_hip"},
                 "nose":{"nose"},
                 "neck":{"neck"},
                 "tail":{"root_of_tail"}
                 }
    elif dataset == 'animal_pose':
        keypoints = ["left_eye","right_eye","nose","left_ear","right_ear","left_shoulder","right_shoulder","left_hip",
                     "right_hip","left_elbow","right_elbow","left_knee","right_knee", "L_F_Paw","R_F_Paw","L_B_Paw",
                     "R_B_Paw","throat","wither","tail"]
        group = {
            "eye":{"left_eye","right_eye"},
            "ear":{"left_ear","right_ear"},
            "paw":{"L_F_Paw","L_B_Paw","R_F_Paw","R_B_Paw"},
            "knee":{"left_elbow", "right_elbow", "left_knee","right_knee"},
            "hip":{"left_shoulder","right_shoulder","left_hip","right_hip"},
            "nose":{"nose"},
            "throat":{"throat"},
            "wither":{"wither"},
            "tail":{"tail"}
        }
    elif dataset == 'tigdog':
        keypoints = ["left_eye", "right_eye", "chin", "L_F_Paw", "R_F_Paw", "L_B_Paw", "R_B_Paw", "root_of_tail",
                     "left_elbow", "right_elbow", "left_knee", "right_knee", "left_high_shoulder","right_high_shoulder",
                     "left_shoulder", "right_shoulder", "left_hip", "right_hip", "neck"]
        group = {
            "eye": {"left_eye", "right_eye"},
            "chin": {"chin"},
            "shoulder": {"left_high_shoulder", "right_high_shoulder"},
            "hip": {"left_hip", "right_hip"},
            "elbow": {"left_shoulder", "right_shoulder"},
            "knee": {"left_elbow", "right_elbow", "left_knee", "right_knee"},
            "hooves": {"L_F_Paw", "L_B_Paw", "R_F_Paw", "R_B_Paw"},
        }

    cocoDt = cocoGt.coco.loadRes(res_file)
    oksThreshold = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    res = {key:{} for key in oksThreshold}
    for ind in range(len(oksThreshold)):
        oks_sum_list = []
        vis_sum_list = []
        mean_oks = []
        oks_list = [[] for i in range(args.num_joints)]
        group_oks_list = [[] for i in range(len(group))]
        num_joints = args.num_joints
        for i in range(num_joints):
            mean_oks.append(0.0)
            oks_sum_list.append(0.0)
            vis_sum_list.append(0)
        for img_id in cocoGt.coco.getImgIds():
            img = cocoGt.coco.loadImgs(img_id)[0]
            ann_ids_gt = cocoGt.coco.getAnnIds(imgIds=img['id'])
            ann_ids_dt = cocoDt.getAnnIds(imgIds=img['id'])
            # get anns for gt and dt
            anns_gt = cocoGt.coco.loadAnns(ann_ids_gt)
            anns_dt = cocoDt.loadAnns(ann_ids_dt)
            # if anns_gt[0]['category_id'] != 2:
            #     continue
            for i in range(len(anns_gt)):
                ann_gt = anns_gt[i]
                for k in range(len(anns_dt)):
                    ann_dt = anns_dt[k]
                    if ann_dt['anno_id'] == ann_gt['id']:
                        oks_sum_list,vis_sum_list = compute_oks(oks_list=oks_list,oks_sum_list=oks_sum_list,vis_sum_list=vis_sum_list,gt=ann_gt,
                                                                dt=ann_dt,threshold=oksThreshold[ind],kpt_oks_sigmas=dataset_definition['sigmas'])
                        # oks_sum_list,vis_sum_list = compute_oks_no_threshold(oks_list=oks_list,oks_sum_list=oks_sum_list,vis_sum_list=vis_sum_list,gt=ann_gt,
                        #                                                      dt=ann_dt,kpt_oks_sigmas=dataset_definition['sigmas'])
                        break
                    else:
                        continue
        # 统计每个点对应的mean_oks
        for i in range(num_joints):
            mean_oks[i] = oks_sum_list[i] / vis_sum_list[i]

        # oks_list 只保存可见点对应的OKS 每个点可见数量不一样
        oks_std = [np.std(sublist) for sublist in oks_list]
        # oks_mean = [np.mean(sublist) for sublist in oks_list]
        oks_dict = {keypoint: {'mean':oks_mean_value,'std':oks_std_value,"sum":oks_sum_value} for keypoint,oks_mean_value,oks_std_value,oks_sum_value in zip(keypoints,mean_oks,oks_std,oks_sum_list)}
        res[oksThreshold[ind]]['keypoints'] = oks_dict

        for keypoint, sublist in zip(keypoints, oks_list):
            for i, part in enumerate(group):
                if keypoint in group[part]:
                    group_oks_list[i] += sublist
                    break

        # group_oks_list中保存了分组中每个关键点对应的所有可见的OKS结果
        # 计算mean和std时，需要考虑被阈值过滤掉的那部分吗
        # 应该将其考虑为0，还是将其考虑为被阈值过滤前原本的值呢
        # 我应该保留原本的值在oks_list中，然后用阈值依次过滤处理
        # 但只是加快了处理速度
        std_group_oks = [np.std(sublist) for sublist in group_oks_list]
        mean_group_oks = [np.mean(sublist) for sublist in
                          group_oks_list]

        res[oksThreshold[ind]]['group'] = {part: {'mean': mean, 'std': std} for part, mean, std in
                                           zip(group, mean_group_oks, std_group_oks)}

    # 统计不同阈值下的平均值
    mAP_mean = {part:[] for part in group}
    mAP_std = {part:[] for part in group}
    for threshold in res:
        for part in res[threshold]['group']:
            mAP_mean[part].append(res[threshold]['group'][part]['mean'])
            mAP_std[part].append(res[threshold]['group'][part]['std'])
    res['average'] = {part:{} for part in group}

    # 出现mAP高的反而可能mAP的std较大的情况，是因为阈值过滤的问题
    # mAP较低。大部分被过滤 -> 大部分为0 -> std较小
    for part in group:
        res['average'][part] = {'mean':np.mean(mAP_mean[part]),'std':np.mean(mAP_std[part])}

    path = f'./statistics_file/{model_name}-{dataset}-{mode}-keypoint-oks.json'
    with open(path,'w') as f:
        json.dump(res,f,indent=4)
    return res


def oks_performance_test_keypoint_no_threshold(args,model_name,dataset="ap_10k",mode="val"):
    # 加载 COCO ground truth 数据
    with open(args.keypoints_path,'r') as f:
        dataset_definition = json.load(f)
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
                transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode,transform=data_transform["test"],
                          fixed_size=args.fixed_size,data_type="keypoints",num_joints=args.num_joints)
    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_results.json'

    if not os.path.exists(res_file):
        json_generate_batch(args,model_name,cocoGt,res_file)

    if dataset == 'ap_10k':
        keypoints = ["left_eye", "right_eye", "nose", "neck", "root_of_tail", "left_shoulder", "left_elbow", "L_F_Paw",
                     "right_shoulder", "right_elbow", "R_F_Paw", "left_hip", "left_knee", "L_B_Paw", "right_hip",
                     "right_knee", "R_B_Paw"]
        group = {"eye":{"left_eye","right_eye"},
                 "paw":{"L_F_Paw","L_B_Paw","R_F_Paw","R_B_Paw"},
                 "knee":{"left_elbow", "right_elbow", "left_knee","right_knee"},
                 "hip":{"left_shoulder","right_shoulder","left_hip","right_hip"},
                 "nose":{"nose"},
                 "neck":{"neck"},
                 "tail":{"root_of_tail"}
                 }
    else:
        keypoints = ["left_eye","right_eye","nose","left_ear","right_ear","left_shoulder","right_shoulder","left_hip",
                     "right_hip","left_elbow","right_elbow","left_knee","right_knee", "L_F_Paw","R_F_Paw","L_B_Paw",
                     "R_B_Paw","throat","wither","tail"]
        group = {
            "eye":{"left_eye","right_eye"},
            "ear":{"left_ear","right_ear"},
            "paw":{"L_F_Paw","L_B_Paw","R_F_Paw","R_B_Paw"},
            "knee":{"left_elbow", "right_elbow", "left_knee","right_knee"},
            "hip":{"left_shoulder","right_shoulder","left_hip","right_hip"},
            "nose":{"nose"},
            "throat":{"throat"},
            "wither":{"wither"},
            "tail":{"tail"}
        }

    ori_path = f'./statistics_file/{model_name}-{dataset}-{mode}-keypoint-oks.json'
    with open(ori_path,'r') as f:
        ori_data = json.load(f)

    ori_data['no_threshold'] = {}

    cocoDt = cocoGt.coco.loadRes(res_file)
    oks_sum_list = []
    vis_sum_list = []
    mean_oks = []
    oks_list = [[] for i in range(args.num_joints)]
    group_oks_list = [[] for i in range(len(group))]
    num_joints = args.num_joints
    for i in range(num_joints):
        mean_oks.append(0.0)
        oks_sum_list.append(0.0)
        vis_sum_list.append(0)
    for img_id in cocoGt.coco.getImgIds():
        img = cocoGt.coco.loadImgs(img_id)[0]
        ann_ids_gt = cocoGt.coco.getAnnIds(imgIds=img['id'])
        ann_ids_dt = cocoDt.getAnnIds(imgIds=img['id'])
        # get anns for gt and dt
        anns_gt = cocoGt.coco.loadAnns(ann_ids_gt)
        anns_dt = cocoDt.loadAnns(ann_ids_dt)

        for i in range(len(anns_gt)):
            ann_gt = anns_gt[i]
            for k in range(len(anns_dt)):
                ann_dt = anns_dt[i]
                if ann_dt['anno_id'] == ann_gt['id']:
                    oks_sum_list,vis_sum_list = compute_oks_no_threshold(oks_list=oks_list,oks_sum_list=oks_sum_list,vis_sum_list=vis_sum_list,gt=ann_gt,
                                                                         dt=ann_dt,kpt_oks_sigmas=dataset_definition['sigmas'])
                    break
                else:
                    continue
    # 统计每个点对应的mean_oks
    for i in range(num_joints):
        mean_oks[i] = oks_sum_list[i] / vis_sum_list[i]

    # oks_list 只保存可见点对应的OKS 每个点可见数量不一样
    oks_std = [np.std(sublist) for sublist in oks_list]
    # oks_mean = [np.mean(sublist) for sublist in oks_list]
    oks_dict = {keypoint: {'mean':oks_mean_value,'std':oks_std_value,"sum":oks_sum_value} for keypoint,oks_mean_value,oks_std_value,oks_sum_value in zip(keypoints,mean_oks,oks_std,oks_sum_list)}
    ori_data['no_threshold']['keypoints'] = oks_dict

    for keypoint, sublist in zip(keypoints, oks_list):
        for i, part in enumerate(group):
            if keypoint in group[part]:
                group_oks_list[i] += sublist
                break

    # group_oks_list中保存了分组中每个关键点对应的所有可见的OKS结果
    # 计算mean和std时，需要考虑被阈值过滤掉的那部分吗
    # 应该将其考虑为0，还是将其考虑为被阈值过滤前原本的值呢
    # 我应该保留原本的值在oks_list中，然后用阈值依次过滤处理
    # 但只是加快了处理速度
    std_group_oks = [np.std(sublist) for sublist in group_oks_list]
    mean_group_oks = [np.mean(sublist) for sublist in group_oks_list]

    ori_data['no_threshold']['group'] = {part: {'mean': mean, 'std': std} for part, mean, std in
                                         zip(group, mean_group_oks, std_group_oks)}

    path = f'./statistics_file/{model_name}-{dataset}-{mode}-keypoint-oks.json'
    with open(path,'a') as f:
        json.dump(ori_data,f,indent=4)
    return ori_data


# 在animal pose dataset的label data上计算模型的Percentage Correct Keypoints指标
def pck_performance_test(args,model_name,dataset="ap_10k",mode="val",threshold=0.05):
    # 加载 COCO ground truth 数据
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode,transform=data_transform["test"],
                          fixed_size=args.fixed_size,data_type="keypoints",num_joints=args.num_joints)
    # test_dataset_loader = DataLoader(cocoGt,
    #                                  batch_size=1,
    #                                  shuffle=False,
    #                                  pin_memory=True,
    #                                  num_workers=8,
    #                                  collate_fn=cocoGt.collate_fn)

    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_results.json'

    if not os.path.exists(res_file):
        json_generate_batch(args,model_name,cocoGt,res_file)

    cocoDt = cocoGt.coco.loadRes(res_file)
    pck_threshold = threshold
    pck_valid_num = 0
    total_valid_num = 0
    for img_id in cocoGt.coco.getImgIds():
        # get the image
        img = cocoGt.coco.loadImgs(img_id)[0]

        # get the GT keypoints
        gt_ann_ids = cocoGt.coco.getAnnIds(imgIds=img['id'],iscrowd=None)
        gt_anns = cocoGt.coco.loadAnns(gt_ann_ids)

        # get the Pred keypoints
        dt_ann_ids = cocoDt.getAnnIds(imgIds=img['id'],iscrowd=None)
        dt_anns = cocoDt.loadAnns(dt_ann_ids)

        for gt_ann in gt_anns:
            for dt_ann in dt_anns:
                if gt_ann['id'] == dt_ann['anno_id']:
                    gt = gt_ann['keypoints']
                    dt = dt_ann['keypoints']
                    # reshape to (17,3)
                    gt = np.array(gt).reshape(args.num_joints,3)
                    dt = np.array(dt).reshape(args.num_joints,3)
                    # compute the normal 欧氏 distance
                    # 这里我使用两眼的间距作归一化参考
                    # 如果缺失眼部数据则采用平均眼距
                    # if gt[0,2] == 0 or gt[1,2] == 0:
                    #     eye_dist = avg_norm_dist
                    # else:
                    #     eye_dist = np.linalg.norm(gt[0,:2]-gt[1,:2])
                    # dist = np.linalg.norm(gt[:,:2]-dt[:,:2],axis=-1) / eye_dist
                    scale = math.sqrt(gt_ann['area'])
                    dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                    # 忽略可见性为0的点
                    mask = gt[:,2] > 0
                    # 计算布尔数组中True值的比例，小于阈值即为True
                    pck_arr = dist[mask] < pck_threshold
                    pck_valid_num += np.sum(pck_arr)
                    total_valid_num += len(pck_arr)
    # compute average pck value
    mean_pck = float(pck_valid_num / total_valid_num)

    output_path = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_performance.txt'
    with open(output_path,'a') as f:
        f.write(f"\nMean PCK on Test Dataset:{mean_pck}\n")

    return mean_pck


# flag = true -> return mean_pck
# flag = false -> return sum_pck_correct_count
def pck_performance_keypoint_test(args,model_name,dataset="ap_10k",mode="val",flag=True,threshold=0.05):
    # 加载 COCO ground truth 数据
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode,transform=data_transform["test"],
                          fixed_size=args.fixed_size,data_type="keypoints",num_joints=args.num_joints)
    # test_dataset_loader = DataLoader(cocoGt,
    #                                  batch_size=1,
    #                                  shuffle=False,
    #                                  pin_memory=True,
    #                                  num_workers=8,
    #                                  collate_fn=cocoGt.collate_fn)

    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_results.json'

    if not os.path.exists(res_file):
        json_generate_batch(args,model_name,cocoGt,res_file)

    cocoDt = cocoGt.coco.loadRes(res_file)
    pck_threshold = threshold
    keypoint_acc_num = [0 for _ in range(args.num_joints)]
    keypoint_visible_num = [0 for _ in range(args.num_joints)]

    for img_id in cocoGt.coco.getImgIds():
        # get the image
        img = cocoGt.coco.loadImgs(img_id)[0]

        # get the GT keypoints
        gt_ann_ids = cocoGt.coco.getAnnIds(imgIds=img['id'],iscrowd=None)
        gt_anns = cocoGt.coco.loadAnns(gt_ann_ids)

        # get the Pred keypoints
        dt_ann_ids = cocoDt.getAnnIds(imgIds=img['id'],iscrowd=None)
        dt_anns = cocoDt.loadAnns(dt_ann_ids)

        for gt_ann in gt_anns:
            for dt_ann in dt_anns:
                if gt_ann['id'] == dt_ann['anno_id']:
                    gt = gt_ann['keypoints']
                    dt = dt_ann['keypoints']
                    # reshape to (17,3)
                    gt = np.array(gt).reshape(args.num_joints,3)
                    dt = np.array(dt).reshape(args.num_joints,3)
                    # compute the normal 欧氏 distance
                    # 这里我使用两眼的间距作归一化参考
                    # 如果缺失眼部数据则采用平均眼距
                    scale = math.sqrt(gt_ann['area'])
                    dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                    # 忽略可见性为0的点
                    mask = gt[:,2] > 0
                    # 计算布尔数组中True值的比例，小于阈值即为True
                    pck_acc = [1 if dist[i] < pck_threshold else 0 for i in range(len(dist))]
                    keypoint_acc_num = [a + b for a,b, in zip(pck_acc,keypoint_acc_num)]
                    keypoint_visible_num = [a + b for a,b in zip(mask,keypoint_visible_num)]

    # compute average pck value
    if flag:
        res_pck = [keypoint_acc_num[i] / keypoint_visible_num[i] for i in range(args.num_joints)]
    else:
        res_pck = keypoint_acc_num

    if dataset == 'ap_10k':
        keypoints = ["left_eye", "right_eye", "nose", "neck", "root_of_tail", "left_shoulder", "left_elbow", "L_F_Paw",
                     "right_shoulder", "right_elbow", "R_F_Paw", "left_hip", "left_knee", "L_B_Paw", "right_hip",
                     "right_knee", "R_B_Paw"]
        group = {"eye":{"left_eye","right_eye"},
                 "paw":{"L_F_Paw","L_B_Paw","R_F_Paw","R_B_Paw"},
                 "knee":{"left_elbow", "right_elbow", "left_knee","right_knee"},
                 "hip":{"left_shoulder","right_shoulder","left_hip","right_hip"},
                 "nose":{"nose"},
                 "neck":{"neck"},
                 "tail":{"root_of_tail"}
                 }
    elif dataset == 'animal_pose':
        keypoints = ["left_eye", "right_eye", "nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                     "left_hip","right_hip","left_elbow", "right_elbow", "left_knee", "right_knee", "L_F_Paw",
                     "R_F_Paw", "L_B_Paw", "R_B_Paw", "throat","wither", "tail"]
        group = {
            "eye":{"left_eye","right_eye"},
            "ear":{"left_ear","right_ear"},
            "paw":{"L_F_Paw","L_B_Paw","R_F_Paw","R_B_Paw"},
            "knee":{"left_elbow", "right_elbow", "left_knee","right_knee"},
            "hip":{"left_shoulder","right_shoulder","left_hip","right_hip"},
            "nose":{"nose"},
            "throat":{"throat"},
            "wither":{"wither"},
            "tail":{"tail"}
        }
    elif dataset == 'tigdog':
        keypoints = ["left_eye", "right_eye", "chin", "L_F_Paw", "R_F_Paw", "L_B_Paw", "R_B_Paw", "root_of_tail",
                     "left_elbow", "right_elbow", "left_knee", "right_knee", "left_high_shoulder",
                     "right_high_shoulder", "left_shoulder", "right_shoulder", "left_hip", "right_hip", "neck"]
        group = {"eye":{"left_eye","right_eye"},
                 "chin":{"chin"},
                 "shoulder":{"left_shoulder","right_shoulder"},
                 "hip":{"left_hip","right_hip"},
                 "elbow":{"left_elbow", "right_elbow"},
                 "knee":{"left_knee","right_knee"},
                 "hoove":{"L_F_Paw","L_B_Paw","R_F_Paw","R_B_Paw"},
                 }
    pck_dict = {keypoint: pck_value for keypoint, pck_value in zip(keypoints, res_pck)}

    path = f'./statistics_file/{model_name}-{dataset}-{mode}-keypoint-pck@{pck_threshold}.json'
    with open(path,'w') as f:
        json.dump(pck_dict,f,indent=4)
    return pck_dict


def pck_performance_group_test(args,model_name,dataset="ap_10k",mode="val",flag=True,threshold=0.05):
    # 加载 COCO ground truth 数据
    data_root = args.val_data_path
    with open(args.keypoints_path,'r') as f:
        animal_kps_info = json.load(f)
    data_transform = {
        "test": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
            # transforms.RandomHorizontalFlip(1, animal_kps_info["flip_pairs"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode,transform=data_transform["test"],
                          fixed_size=args.fixed_size,data_type="keypoints",num_joints=args.num_joints)
    # test_dataset_loader = DataLoader(cocoGt,
    #                                  batch_size=1,
    #                                  shuffle=False,
    #                                  pin_memory=True,
    #                                  num_workers=8,
    #                                  collate_fn=cocoGt.collate_fn)

    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_results.json'

    if not os.path.exists(res_file):
        json_generate_batch(args,model_name,cocoGt,res_file)

    cocoDt = cocoGt.coco.loadRes(res_file)
    pck_threshold = threshold
    category_list = [cocoGt.coco.cats[key]['id'] for key in cocoGt.coco.cats]
    category_res = []
    for animal_category in category_list:
        keypoint_acc_num = [0 for i in range(args.num_joints)]
        keypoint_visible_num = [0 for i in range(args.num_joints)]

        for img_id in cocoGt.coco.getImgIds():
            # get the image
            img = cocoGt.coco.loadImgs(img_id)[0]

            # get the GT keypoints
            gt_ann_ids = cocoGt.coco.getAnnIds(imgIds=img['id'],iscrowd=None)
            gt_anns = cocoGt.coco.loadAnns(gt_ann_ids)

            # get the Pred keypoints
            dt_ann_ids = cocoDt.getAnnIds(imgIds=img['id'],iscrowd=None)
            dt_anns = cocoDt.loadAnns(dt_ann_ids)

            for gt_ann in gt_anns:
                if gt_ann['category_id'] != animal_category:
                    break
                for dt_ann in dt_anns:
                    if gt_ann['id'] == dt_ann['anno_id']:
                        gt = gt_ann['keypoints']
                        dt = dt_ann['keypoints']

                        # img_path = os.path.join(args.val_data_path,img['file_name'])
                        # gt_x = gt[0::3]
                        # gt_y = gt[1::3]
                        # dt_x = dt[0::3]
                        # dt_y = dt[1::3]
                        # im = cv2.imread(img_path)
                        # # 绘制标注点
                        # for x, y in zip(gt_x, gt_y):
                        #     cv2.circle(im, (int(x), int(y)), 3, (255, 0, 0), -1)  # 在图像上绘制绿色圆形
                        #
                        # for x, y in zip(dt_x, dt_y):
                        #     cv2.circle(im, (int(x), int(y)), 3, (0, 0, 255), -1)  # 在图像上绘制红色圆形
                        #
                        # wname = 'image'
                        # cv2.imshow(wname, im)
                        # cv2.waitKey()

                        # reshape to (17,3)
                        gt = np.array(gt).reshape(args.num_joints,3)
                        dt = np.array(dt).reshape(args.num_joints,3)
                        # compute the normal 欧氏 distance
                        # 这里我使用两眼的间距作归一化参考
                        # 如果缺失眼部数据则采用平均眼距
                        scale = math.sqrt(gt_ann['area'])
                        dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                        # 忽略可见性为0的点
                        mask = gt[:,2] > 0
                        # 计算布尔数组中True值的比例，小于阈值即为True
                        pck_acc = [1 if dist[i] < pck_threshold else 0 for i in range(len(dist))]
                        keypoint_acc_num = [a + b for a,b, in zip(pck_acc,keypoint_acc_num)]
                        keypoint_visible_num = [a + b for a,b in zip(mask,keypoint_visible_num)]

        # compute average pck value
        if flag:
            res_pck = [keypoint_acc_num[i] / keypoint_visible_num[i] for i in range(args.num_joints)]
        else:
            res_pck = keypoint_acc_num

        if dataset == 'ap_10k':
            keypoints = ["left_eye", "right_eye", "nose", "neck", "root_of_tail", "left_shoulder", "left_elbow", "L_F_Paw",
                         "right_shoulder", "right_elbow", "R_F_Paw", "left_hip", "left_knee", "L_B_Paw", "right_hip",
                         "right_knee", "R_B_Paw"]
            group = {"eye":{"left_eye","right_eye"},
                     "paw":{"L_F_Paw","L_B_Paw","R_F_Paw","R_B_Paw"},
                     "knee":{"left_elbow", "right_elbow", "left_knee","right_knee"},
                     "hip":{"left_shoulder","right_shoulder","left_hip","right_hip"},
                     "nose":{"nose"},
                     "neck":{"neck"},
                     "tail":{"root_of_tail"}
                     }
        elif dataset == 'animal_pose':
            keypoints = ["left_eye", "right_eye", "nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                         "left_hip","right_hip","left_elbow", "right_elbow", "left_knee", "right_knee", "L_F_Paw",
                         "R_F_Paw", "L_B_Paw", "R_B_Paw", "throat","wither", "tail"]
            group = {
                "eye":{"left_eye","right_eye"},
                "ear":{"left_ear","right_ear"},
                "paw":{"L_F_Paw","L_B_Paw","R_F_Paw","R_B_Paw"},
                "knee":{"left_elbow", "right_elbow", "left_knee","right_knee"},
                "hip":{"left_shoulder","right_shoulder","left_hip","right_hip"},
                "nose":{"nose"},
                "throat":{"throat"},
                "wither":{"wither"},
                "tail":{"tail"}
            }
        elif dataset == 'tigdog':
            keypoints = ["left_eye", "right_eye", "chin", "L_F_Paw", "R_F_Paw", "L_B_Paw", "R_B_Paw", "root_of_tail",
                         "left_elbow", "right_elbow", "left_knee", "right_knee", "left_high_shoulder",
                         "right_high_shoulder", "left_shoulder", "right_shoulder", "left_hip", "right_hip", "neck"]
            group = {
                "eye": {"left_eye", "right_eye"},
                "chin": {"chin"},
                "shoulder": {"left_high_shoulder", "right_high_shoulder"},
                "hip": {"left_hip", "right_hip"},
                "elbow": {"left_shoulder","right_shoulder"},
                "knee": {"left_elbow", "right_elbow","left_knee", "right_knee"},
                "hooves": {"L_F_Paw", "L_B_Paw", "R_F_Paw", "R_B_Paw"},
                }
        pck_kps_dict = {keypoint: {"acc":acc_value,"vis":vis_value,"pck":acc_value/vis_value} for keypoint, acc_value, vis_value in zip(keypoints, keypoint_acc_num,keypoint_visible_num)}
        group_kps_dict = {part: {"acc":0,"vis":0} for part in group}
        for keypoint in keypoints:
            for part in group:
                if keypoint in group[part]:
                    group_kps_dict[part]['acc'] += pck_kps_dict[keypoint]['acc']
                    group_kps_dict[part]['vis'] += pck_kps_dict[keypoint]['vis']
                    break

        group_dict = {part:(group_kps_dict[part]['acc'] / group_kps_dict[part]['vis']) for part in group_kps_dict}
        acc_num = 0
        all_num = 0
        for part in group_kps_dict:
            acc_num += group_kps_dict[part]['acc']
            all_num += group_kps_dict[part]['vis']
        mean_pck = acc_num / all_num
        group_dict['mean'] = mean_pck
        if animal_category == 1:
            print("horse")
        else:
            print("tiger")
        print(group_dict)
        print(pck_kps_dict)
        category_res.append(group_dict)

    return category_res


def pck_performance_test_image_level(args,model_name,dataset="ap_10k",mode="val",threshold=0.05):
    # 加载 COCO ground truth 数据
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode,transform=data_transform["test"],
                          fixed_size=args.fixed_size,data_type="keypoints",num_joints=args.num_joints)
    # test_dataset_loader = DataLoader(cocoGt,
    #                                  batch_size=1,
    #                                  shuffle=False,
    #                                  pin_memory=True,
    #                                  num_workers=8,
    #                                  collate_fn=cocoGt.collate_fn)

    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_results.json'

    if not os.path.exists(res_file):
        json_generate_batch(args,model_name,cocoGt,res_file)

    cocoDt = cocoGt.coco.loadRes(res_file)
    pck_threshold = threshold

    # image level pck
    pck_list = []
    for img_id in cocoGt.coco.getImgIds():
        # get the image
        img = cocoGt.coco.loadImgs(img_id)[0]

        # get the GT keypoints
        gt_ann_ids = cocoGt.coco.getAnnIds(imgIds=img['id'])
        gt_anns = cocoGt.coco.loadAnns(gt_ann_ids)

        # get the Pred keypoints
        dt_ann_ids = cocoDt.getAnnIds(imgIds=img['id'])
        dt_anns = cocoDt.loadAnns(dt_ann_ids)

        for gt_ann in gt_anns:
            for dt_ann in dt_anns:
                if gt_ann['id'] == dt_ann['anno_id']:
                    gt = gt_ann['keypoints']
                    dt = dt_ann['keypoints']
                    # reshape to (17,3)
                    gt = np.array(gt).reshape(args.num_joints,3)
                    dt = np.array(dt).reshape(args.num_joints,3)
                    # compute the normal 欧氏 distance
                    # 这里我使用两眼的间距作归一化参考
                    # 如果缺失眼部数据则采用平均眼距
                    # if gt[0,2] == 0 or gt[1,2] == 0:
                    #     eye_dist = avg_norm_dist
                    # else:
                    #     eye_dist = np.linalg.norm(gt[0,:2]-gt[1,:2])
                    # dist = np.linalg.norm(gt[:,:2]-dt[:,:2],axis=-1) / eye_dist
                    scale = math.sqrt(gt_ann['area'])
                    dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                    # 忽略可见性为0的点
                    mask = gt[:,2] > 0
                    # 计算布尔数组中True值的比例，小于阈值即为True
                    pck_arr = dist[mask] < pck_threshold
                    pck_list.append(np.sum(pck_arr) / len(pck_arr))

    # compute average pck value
    mean_pck = np.mean(pck_list)
    output_path = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_performance.txt'
    with open(output_path,'a') as f:
        f.write(f"\nMean PCK on Test Dataset:{mean_pck}\n")
    print(f"Mean PCK on Test Dataset:{mean_pck}")
    return mean_pck


def pck_performance_animal(args,model_name,dataset="ap_10k",mode="val",threshold=0.05):
    # 加载 COCO ground truth 数据
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode, transform=data_transform["test"],
                          fixed_size=args.fixed_size, data_type="keypoints", num_joints=args.num_joints)
    # test_dataset_loader = DataLoader(cocoGt,
    #                                  batch_size=1,
    #                                  shuffle=False,
    #                                  pin_memory=True,
    #                                  num_workers=8,
    #                                  collate_fn=cocoGt.collate_fn)

    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_results.json'

    if not os.path.exists(res_file):
        json_generate_batch(args, model_name, cocoGt, res_file)

    cocoDt = cocoGt.coco.loadRes(res_file)
    pck_threshold = threshold

    # animal - image level pck
    pck_list = [[] for _ in range(len(cocoGt.coco.cats))]
    for img_id in cocoGt.coco.getImgIds():
        # get the image
        img = cocoGt.coco.loadImgs(img_id)[0]

        # get the GT keypoints
        gt_ann_ids = cocoGt.coco.getAnnIds(imgIds=img['id'])
        gt_anns = cocoGt.coco.loadAnns(gt_ann_ids)

        # get the Pred keypoints
        dt_ann_ids = cocoDt.getAnnIds(imgIds=img['id'])
        dt_anns = cocoDt.loadAnns(dt_ann_ids)

        for gt_ann in gt_anns:
            for dt_ann in dt_anns:
                if gt_ann['id'] == dt_ann['anno_id']:
                    category = gt_ann['category_id']
                    gt = gt_ann['keypoints']
                    dt = dt_ann['keypoints']
                    # reshape to (17,3)
                    gt = np.array(gt).reshape(args.num_joints, 3)
                    dt = np.array(dt).reshape(args.num_joints, 3)
                    # compute the normal 欧氏 distance
                    # 这里我使用两眼的间距作归一化参考
                    # 如果缺失眼部数据则采用平均眼距
                    # if gt[0,2] == 0 or gt[1,2] == 0:
                    #     eye_dist = avg_norm_dist
                    # else:
                    #     eye_dist = np.linalg.norm(gt[0,:2]-gt[1,:2])
                    # dist = np.linalg.norm(gt[:,:2]-dt[:,:2],axis=-1) / eye_dist
                    scale = math.sqrt(gt_ann['area'])
                    dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                    # 忽略可见性为0的点
                    mask = gt[:, 2] > 0
                    # 计算布尔数组中True值的比例，小于阈值即为True
                    pck_arr = dist[mask] < pck_threshold
                    pck_list[category-1].append(np.sum(pck_arr) / len(pck_arr))

    # compute average pck value
    mean_pck = [np.mean(sublist) if len(sublist) > 0 else 0.0 for sublist in pck_list]
    category_info = cocoGt.coco.cats
    animals = [category_info[cat]['name'] for cat in category_info]
    mean_pck_dict = {"pck":{name:val for name,val in zip(animals,mean_pck)}}
    write_path = f'./statistics_file/{model_name}_{dataset}_{mode}_animal_results.json'
    with open(write_path,'w') as f:
        json.dump(mean_pck_dict,f,indent=4)
    return mean_pck_dict


def nme_performance_test(args,model_name,dataset="ap_10k",mode="val"):
    # 加载 COCO ground truth 数据
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode, transform=data_transform["test"],
                          fixed_size=args.fixed_size, data_type="keypoints", num_joints=args.num_joints)
    # test_dataset_loader = DataLoader(cocoGt,
    #                                  batch_size=1,
    #                                  shuffle=False,
    #                                  pin_memory=True,
    #                                  num_workers=8,
    #                                  collate_fn=cocoGt.collate_fn)

    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_results.json'

    if not os.path.exists(res_file):
        json_generate_batch(args, model_name, cocoGt, res_file)

    cocoDt = cocoGt.coco.loadRes(res_file)

    total_nme = 0
    nme_count = 0
    for img_id in cocoGt.coco.getImgIds():
        # get the image
        img = cocoGt.coco.loadImgs(img_id)[0]

        # get the GT keypoints
        gt_ann_ids = cocoGt.coco.getAnnIds(imgIds=img['id'],iscrowd=None)
        gt_anns = cocoGt.coco.loadAnns(gt_ann_ids)

        # get the Pred keypoints
        dt_ann_ids = cocoDt.getAnnIds(imgIds=img['id'],iscrowd=None)
        dt_anns = cocoDt.loadAnns(dt_ann_ids)

        for gt_ann in gt_anns:
            for dt_ann in dt_anns:
                if gt_ann['id'] == dt_ann['anno_id']:
                    gt = gt_ann['keypoints']
                    dt = dt_ann['keypoints']
                    scale = math.sqrt(gt_ann['area'])
                    # reshape to (17,3)
                    gt = np.array(gt).reshape(args.num_joints,3)
                    dt = np.array(dt).reshape(args.num_joints,3)
                    nme_weight = 100
                    nme = compute_nme(gt, dt, scale=scale) * nme_weight

                    # 累加到累积NME中
                    total_nme += nme
                    nme_count += 1

    mean_nme = total_nme / nme_count
    return mean_nme


# flag = true -> return mean_nme
# flag = false -> return sum_nme
def nme_performance_keypoint_test(args,model_name,dataset="ap_10k",mode="val",flag=True):
    # 加载 COCO ground truth 数据
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode, transform=data_transform["test"],
                          fixed_size=args.fixed_size, data_type="keypoints", num_joints=args.num_joints)
    # test_dataset_loader = DataLoader(cocoGt,
    #                                  batch_size=1,
    #                                  shuffle=False,
    #                                  pin_memory=True,
    #                                  num_workers=8,
    #                                  collate_fn=cocoGt.collate_fn)

    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_results.json'

    if not os.path.exists(res_file):
        json_generate_batch(args, model_name, cocoGt, res_file)

    cocoDt = cocoGt.coco.loadRes(res_file)

    total_nme = [0.0 for i in range(args.num_joints)]
    nme_count = [0 for i in range(args.num_joints)]
    nme_list = [[] for i in range(args.num_joints)]
    nme_weight = 1

    for img_id in cocoGt.coco.getImgIds():
        # get the image
        img = cocoGt.coco.loadImgs(img_id)[0]

        # get the GT keypoints
        gt_ann_ids = cocoGt.coco.getAnnIds(imgIds=img['id'],iscrowd=None)
        gt_anns = cocoGt.coco.loadAnns(gt_ann_ids)

        # get the Pred keypoints
        dt_ann_ids = cocoDt.getAnnIds(imgIds=img['id'],iscrowd=None)
        dt_anns = cocoDt.loadAnns(dt_ann_ids)

        for gt_ann in gt_anns:
            for dt_ann in dt_anns:
                if gt_ann['id'] == dt_ann['anno_id']:
                    gt = gt_ann['keypoints']
                    dt = dt_ann['keypoints']
                    scale = math.sqrt(gt_ann['area'])
                    # reshape to (17,3)
                    gt = np.array(gt).reshape(args.num_joints,3)
                    dt = np.array(dt).reshape(args.num_joints,3)
                    nme,visible = compute_keypoint_nme(gt, dt, scale=scale)
                    nme *= nme_weight
                    for i in range(args.num_joints):
                        nme_list[i].append(nme[i])
                    # 累加到累积NME中
                    total_nme += nme
                    nme_count += visible

    if flag:
        mean_nme = [nme_val / count for nme_val,count in zip(total_nme,nme_count)]
    else:
        mean_nme = total_nme

    if dataset == 'ap_10k':
        keypoints = ["left_eye", "right_eye", "nose", "neck", "root_of_tail", "left_shoulder", "left_elbow", "L_F_Paw",
                     "right_shoulder", "right_elbow", "R_F_Paw", "left_hip", "left_knee", "L_B_Paw", "right_hip",
                     "right_knee", "R_B_Paw"]
        group = {"eye":{"left_eye","right_eye"},
                 "paw":{"L_F_Paw","L_B_Paw","R_F_Paw","R_B_Paw"},
                 "knee":{"left_elbow", "right_elbow", "left_knee","right_knee"},
                 "hip":{"left_shoulder","right_shoulder","left_hip","right_hip"},
                 "nose":{"nose"},
                 "neck":{"neck"},
                 "tail":{"root_of_tail"}
                 }
    else:
        keypoints = ["left_eye","right_eye","nose","left_ear","right_ear","left_shoulder","right_shoulder","left_hip",
                     "right_hip","left_elbow","right_elbow","left_knee","right_knee", "L_F_Paw","R_F_Paw","L_B_Paw",
                     "R_B_Paw","throat","wither","tail"]
        group = {
            "eye":{"left_eye","right_eye"},
            "ear":{"left_ear","right_ear"},
            "paw":{"L_F_Paw","L_B_Paw","R_F_Paw","R_B_Paw"},
            "knee":{"left_elbow", "right_elbow", "left_knee","right_knee"},
            "hip":{"left_shoulder","right_shoulder","left_hip","right_hip"},
            "nose":{"nose"},
            "throat":{"throat"},
            "wither":{"wither"},
            "tail":{"tail"}
        }

    nme_dict = {}
    nme_list = [[x for x in sublist if x != 0] for sublist in nme_list]
    std_nme = [np.std(sublist) for sublist in nme_list]
    nme_dict['keypoints'] = {keypoint: {'sum':mean,'std':std} for keypoint,mean,std in zip(keypoints,mean_nme,std_nme)}

    group_list = [[] for i in range(len(group))]
    for keypoint,sublist in zip(keypoints,nme_list):
        for i,part in enumerate(group):
            if keypoint in group[part]:
                group_list[i] += sublist
                break

    std_group_nme = [np.std(sublist) for sublist in group_list]
    mean_group_nme = [np.mean(sublist) for sublist in group_list]

    nme_dict['group'] = {part: {'mean':mean,'std':std} for part,mean,std in zip(group,mean_group_nme,std_group_nme)}

    path = f'./statistics_file/{model_name}-{dataset}-{mode}-keypoint-nme@{nme_weight}.json'
    print(path)
    with open(path,'w') as f:
        json.dump(nme_dict,f,indent=4)

    return nme_dict


def eval_model_oks_key(args,model,name,dataset="ap_10k",mode="val",key="model"):
    # 加载 COCO ground truth 数据
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
                transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode,transform=data_transform["test"],
                          fixed_size=args.fixed_size,data_type="keypoints",num_joints=args.num_joints)
    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{name}_{dataset}_{mode}_results.json'

    if not os.path.exists(res_file):
        json_generate_key_batch(args, model, cocoGt, key, res_file)

    cocoDt = cocoGt.coco.loadRes(res_file)
    coco_eval = COCOeval(cocoGt=cocoGt.coco,cocoDt=cocoDt,iouType='keypoints')

    coco_eval.params.useSegm = None
    with open(args.keypoints_path, 'r') as f:
        kps_definition = json.load(f)
    coco_eval.params.kpt_oks_sigmas = np.array(kps_definition['sigmas'])
    coco_eval.evaluate()
    coco_eval.accumulate()

    summary_output = f'{args.output_dir}/results/{name}_{dataset}_{mode}_performance.txt'
    with open(summary_output,'a') as f:
        f.write("Object Keypoints Similarity Performance:\n")
        sys.stdout = f
        coco_eval.summarize()
        sys.stdout = sys.__stdout__
    return coco_eval.stats[0]


def eval_model_pck_key(args,model,name,dataset="ap_10k",mode="val",key="model",threshold=0.05):
    # 加载 COCO ground truth 数据
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=args.fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode,transform=data_transform["test"],
                          fixed_size=args.fixed_size,data_type="keypoints",num_joints=args.num_joints)
    # test_dataset_loader = DataLoader(cocoGt,
    #                                  batch_size=1,
    #                                  shuffle=False,
    #                                  pin_memory=True,
    #                                  num_workers=8,
    #                                  collate_fn=cocoGt.collate_fn)

    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{name}_{dataset}_{mode}_results.json'

    if not os.path.exists(res_file):
        json_generate_key_batch(args,model,cocoGt,key,res_file)

    cocoDt = cocoGt.coco.loadRes(res_file)
    pck_threshold = threshold

    # image level pck
    pck_list = []
    for img_id in cocoGt.coco.getImgIds():
        # get the image
        img = cocoGt.coco.loadImgs(img_id)[0]

        # get the GT keypoints
        gt_ann_ids = cocoGt.coco.getAnnIds(imgIds=img['id'])
        gt_anns = cocoGt.coco.loadAnns(gt_ann_ids)

        # get the Pred keypoints
        dt_ann_ids = cocoDt.getAnnIds(imgIds=img['id'])
        dt_anns = cocoDt.loadAnns(dt_ann_ids)

        for gt_ann in gt_anns:
            for dt_ann in dt_anns:
                if gt_ann['id'] == dt_ann['anno_id']:
                    gt = gt_ann['keypoints']
                    dt = dt_ann['keypoints']
                    # reshape to (17,3)
                    gt = np.array(gt).reshape(args.num_joints,3)
                    dt = np.array(dt).reshape(args.num_joints,3)
                    # compute the normal 欧氏 distance
                    # 这里我使用两眼的间距作归一化参考
                    # 如果缺失眼部数据则采用平均眼距
                    # if gt[0,2] == 0 or gt[1,2] == 0:
                    #     eye_dist = avg_norm_dist
                    # else:
                    #     eye_dist = np.linalg.norm(gt[0,:2]-gt[1,:2])
                    # dist = np.linalg.norm(gt[:,:2]-dt[:,:2],axis=-1) / eye_dist
                    scale = math.sqrt(gt_ann['area'])
                    dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                    # 忽略可见性为0的点
                    mask = gt[:,2] > 0
                    # 计算布尔数组中True值的比例，小于阈值即为True
                    pck_arr = dist[mask] < pck_threshold
                    pck_list.append(np.sum(pck_arr) / len(pck_arr))

    # compute average pck value
    mean_pck = np.mean(pck_list)
    output_path = f'{args.output_dir}/results/{name}_{dataset}_{mode}_performance.txt'
    with open(output_path,'a') as f:
        f.write(f"\nMean PCK on Test Dataset:{mean_pck}\n")
    print(f"Mean PCK on Test Dataset:{mean_pck}")
    return mean_pck


def eval_group_pck_no_flip(args,model_name,kp_index,dataset="ap_10k",mode="val",threshold=0.05):
    if len(kp_index) == 0:
        return 0
    info_path = args.keypoints_path
    with open(info_path,'r') as f:
        info = json.load(f)

    # 加载 COCO ground truth 数据
    data_root = args.val_data_path
    data_transform = {
        "test": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=(256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    cocoGt = CocoKeypoint(root=data_root, dataset=dataset, mode=mode,transform=data_transform["test"],
                          data_type="keypoints",num_joints=args.num_joints)
    # test_dataset_loader = DataLoader(cocoGt,
    #                                  batch_size=1,
    #                                  shuffle=False,
    #                                  pin_memory=True,
    #                                  num_workers=8,
    #                                  collate_fn=cocoGt.collate_fn)

    # 如果本地已经有预测结果  不再重复预测
    res_file = f'{args.output_dir}/results/{model_name}_{dataset}_{mode}_results_no_flip.json'

    if not os.path.exists(res_file):
        json_generate_batch_no_flip(args,model_name,cocoGt,res_file)

    cocoDt = cocoGt.coco.loadRes(res_file)
    pck_threshold = threshold
    keypoint_acc_num = [0 for _ in range(args.num_joints)]
    keypoint_visible_num = [0 for _ in range(args.num_joints)]

    for img_id in cocoGt.coco.getImgIds():
        # get the image
        img = cocoGt.coco.loadImgs(img_id)[0]

        # get the GT keypoints
        gt_ann_ids = cocoGt.coco.getAnnIds(imgIds=img['id'],iscrowd=None)
        gt_anns = cocoGt.coco.loadAnns(gt_ann_ids)

        # get the Pred keypoints
        dt_ann_ids = cocoDt.getAnnIds(imgIds=img['id'],iscrowd=None)
        dt_anns = cocoDt.loadAnns(dt_ann_ids)

        for gt_ann in gt_anns:
            for dt_ann in dt_anns:
                if gt_ann['id'] == dt_ann['anno_id']:
                    gt = gt_ann['keypoints']
                    dt = dt_ann['keypoints']
                    # reshape to (17,3)
                    gt = np.array(gt).reshape(args.num_joints,3)
                    dt = np.array(dt).reshape(args.num_joints,3)
                    # compute the normal 欧氏 distance
                    # 这里我使用两眼的间距作归一化参考
                    # 如果缺失眼部数据则采用平均眼距
                    scale = math.sqrt(gt_ann['area'])
                    dist = np.linalg.norm(gt[:, :2] - dt[:, :2], axis=-1) / scale
                    # 忽略可见性为0的点
                    mask = gt[:,2] > 0
                    # 计算布尔数组中True值的比例，小于阈值即为True
                    pck_acc = [1 if dist[i] < pck_threshold else 0 for i in range(len(dist))]
                    keypoint_acc_num = [a + b for a,b, in zip(pck_acc,keypoint_acc_num)]
                    keypoint_visible_num = [a + b for a,b in zip(mask,keypoint_visible_num)]

    keypoints = info['keypoints']
    total_dict = {}
    for list_index,ind_list in enumerate(kp_index):
        if len(ind_list) > 0:
            # compute average pck value
            group_kps = []
            group_acc_num = []
            group_visible_num = []
            for ind in ind_list:
                group_kps.append(keypoints[ind])
                group_acc_num.append(keypoint_acc_num[ind])
                group_visible_num.append(keypoint_visible_num[ind])
            res_pck = [group_acc_num[i] / group_visible_num[i] for i in range(len(ind_list))]
            pck_dict = {keypoint: pck_value for keypoint, pck_value in zip(group_kps, res_pck)}
            pck_dict['mean'] = np.sum(group_acc_num) / np.sum(group_visible_num)
            print(pck_dict)
            total_dict[list_index] = pck_dict
        else:
            pck_dict = {'mean':0.0}
            print(pck_dict)
            total_dict[list_index] = pck_dict
    path = f'{args.output_dir}/statistics_file/part_results/{model_name}-{dataset}-{mode}-keypoint-pck-no-flip.json'
    with open(path,'w') as f:
        json.dump(total_dict,f,indent=4)


def evaluate_loss(args,model,data_loader):
    model.eval()
    losses = AverageMeter()
    criterion = KpLossLabel()
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    with torch.no_grad():
        pbar = tqdm(range(len(data_loader)))
        for images, targets in data_loader:
            # 将图片传入指定设备device
            images = images.to(args.device)
            # inference
            outputs = model(images)

            target_heatmaps = torch.stack([t["heatmap"].to(args.device) for t in targets])
            target_visibles = torch.stack([torch.tensor(t["visible"]) for t in targets])

            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

            loss = criterion(outputs,target_heatmaps,target_visibles,args)
            losses.update(loss.item())
            pbar.set_description(f"Val Loss: {losses.avg:.7f}. Testing")
            pbar.update()
        pbar.close()
    # print("Average Loss on Val Dataset:",losses.avg)
    return losses.avg


def evaluate(args,model_name,dataset="ap_10k",mode="val"):
    if dataset == "ap_10k":
        mean_oks = oks_performance_test(args,model_name,dataset,mode)
        mean_pck = pck_performance_test_image_level(args,model_name,dataset,mode)

        return mean_oks,mean_pck

    elif dataset == "animal_pose":
        mean_oks = oks_performance_test(args,model_name,dataset,mode)
        mean_pck = pck_performance_test_image_level(args,model_name,dataset,mode)
        return mean_oks,mean_pck

    elif dataset == "tigdog":
        mean_oks = oks_performance_test(args,model_name,dataset,mode)
        mean_pck = pck_performance_test_image_level(args,model_name,dataset,mode)

        return mean_oks,mean_pck


def eval_model(args,model,name,dataset="ap_10k",mode="val"):
    mean_oks = eval_model_oks(args,model,name,dataset,mode)
    mean_pck = eval_model_pck(args,model,name,dataset,mode)
    return mean_oks,mean_pck


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--workers', default=4, type=int, help='number of workers for DataLoader')
    parser.add_argument('--fixed-size', default=[256, 256], nargs='+', type=int, help='input size')
    parser.add_argument('--val-data-path', default='../dataset/animal_pose',type=str, help='data path')
    parser.add_argument('--batch-size', default=32, type=int, help='train batch size')
    parser.add_argument('--device', default='cuda:0',type=str, help='device info')
    parser.add_argument('--output_dir', default='.', type=str, help='data path')
    parser.add_argument('--num-joints', default=19, type=int, help='num_joints')
    parser.add_argument('--keypoints-path', default="./info/animal_pose_keypoints_format.json", type=str,
                        help='dataset_keypoints_format json path')
    args = parser.parse_args()
    # dist = avg_eye_dist(args)

    # import json
    #
    # with open('./results/model-70_val_results.json', 'r') as f:
    #     data = json.load(f)
    #
    # # Sort by id field
    # sorted_data = sorted(data, key=lambda x: int(x['id']))
    #
    # # Check for duplicates
    # ids = set()
    # for d in sorted_data:
    #     if d['id'] in ids:
    #         print(f"Duplicate id found: {d['id']}")
    #     else:
    #         ids.add(d['id'])

    # out_path = './eval_log.txt'
    # model_name = 'model-190'
    # model_name = 'model-ap-10k-aug-best'
    # mean_oks,mean_pck = evaluate(args,model_name,dataset="ap_10k",mode="test")
    # model_name = 'model-0'
    # mean_oks,mean_pck = evaluate(args,model_name,dataset="tigdog",mode="val")
    # print(mean_oks,mean_pck)
    # evaluate(args=args,model_name='model-ap-10k-ori-best',dataset='ap_10k',mode='test')
    # oks,pck = evaluate(args,f"model-{index}",dist,dataset="val")
    # eval_info = f"model-{index}:oks={oks},pck={pck}"
    # print(eval_info)
    # with open(out_path, 'a') as f:
    #     f.write(eval_info + '\n')
    # evaluate(args,model_name='model-251',dataset='ap_10k',mode='train')
    # oks_performance_test_keypoint(args=args,model_name='model-ap-10k-ori-best',dataset='ap_10k',mode='test')
    # model_name = 'model-42'
    # evaluate(args,model_name,dataset="animal_pose",mode="val")
    # model_name = 'model-tigdog'
    # pck_performance_group_test(args,model_name,dataset='tigdog',mode='val')
    # model_name = 'model-119'
    # pck_performance_keypoint_test(args,model_name,dataset='animal_pose',mode='val')
    # oks_performance_test_keypoint(args=args,model_name=model_name,dataset='tigdog',mode='val')
    model_name = 'tiger-183'
    dataset = 'tigdog_tiger'
    mode = 'val'
    if dataset == 'tigdog':
        args.val_data_path = '../../dataset/tigdog'
        args.keypoints_path = '../info/tigdog_keypoints_format.json'
        args.num_joints = 19
    elif dataset == 'ap_10k':
        args.val_data_path = '../../dataset/ap_10k'
        args.keypoints_path = '../info/ap_10k_keypoints_format.json'
        args.num_joints = 17
    elif dataset == 'animal_pose':
        args.val_data_path = '../../dataset/animal_pose'
        args.keypoints_path = '../info/animal_pose_keypoints_format.json'
        args.num_joints = 20

    args.val_data_path = '../../dataset/tigdog_tiger'
    args.keypoints_path = '../info/tigdog_tiger_keypoints_format.json'
    args.num_joints = 19

    oks_performance_test(args, model_name, dataset, mode)
    pck_performance_test_image_level(args, model_name, dataset, mode)

    # pck_performance_keypoint_test(args,model_name,dataset,mode)
    # evaluate(args,model_name,dataset,mode)
    # pck_performance_group_test(args,model_name,dataset='tigdog',mode='val')
    # pck_performance_animal(args,model_name,dataset,mode)
    # oks_performance_test(args,model_name,dataset,mode)
    # pck_performance_test(args,model_name,dataset,mode)
