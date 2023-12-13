import datetime
import logging
import argparse
import os
import random
import json
import numpy as np
import torch
import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader,SequentialSampler
from train_utils.transforms import get_max_preds
from models.hrnet import HighResolutionNet
from train_utils import transforms
from train_utils.dataset import CocoKeypoint
from outer_tools.lib.config import cfg,update_config
from outer_tools.lib.utils.utils import create_logger
from torch.backends import cudnn as cudnn
import math


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main(cfg,args):
    # 将args转换为字典
    args_dict = vars(args)
    # 打印参数值
    logger.info('Args: %s', args_dict)

    if args.seed is not None:
        set_seed(args)
    #

    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    fixed_size = args.fixed_size

    data_transform = {
        "val": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    data_root = args.data_root
    val_dataset = CocoKeypoint(root=data_root, dataset="ap_10k",mode="val",transform=data_transform["val"],
                               fixed_size=args.fixed_size, data_type="keypoints")

    batch_size = args.batch_size
    nw = args.workers  # number of workers
    logger.info('Using %g dataloader workers' % nw)
    #
    base_weight_path = args.pretrained_model_path
    weight_name = args.pretrained_weights_name
    pretrained_weights_path = os.path.join(base_weight_path,weight_name)
    model = HighResolutionNet(num_joints=args.num_joints)
    checkpoint = torch.load(pretrained_weights_path)

    attr_flag = False
    for key in ['state_dict','model','student_model','directly']:
        if key in checkpoint:
            model.load_state_dict(checkpoint[key])
            attr_flag = True
            break
    if not attr_flag:
        model.load_state_dict(checkpoint)

    logger.info(f"model loaded from {pretrained_weights_path}:{key}")

    model = torch.nn.DataParallel(model,device_ids=args.gpus).cuda()
    model.eval()

    data_loader = DataLoader(val_dataset,
                             batch_size=batch_size,
                             sampler=SequentialSampler(val_dataset),
                             pin_memory=True,
                             num_workers=nw,
                             drop_last=False,
                             collate_fn=val_dataset.collate_fn)

    dt_recall_num = np.zeros(17)
    gt_total_num = np.zeros(17)

    dt_confident_num = np.zeros(17)
    dt_confident_acc_num = np.zeros(17)
    # precision = dt_confident_acc_num / dt_confident_num
    # recall = dt_confident_num / gt_num

    with torch.no_grad():
        for imgs,targets in tqdm(data_loader):
            batch_size = imgs.shape[0]
            imgs = imgs.cuda()
            outputs = model(imgs)
            flipped_images = transforms.flip_images(imgs)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5
            # decode keypoint
            pck_threshold = 0.05
            pl_confidence_threshold = 0.4
            reverse_trans = [t["reverse_trans"] for t in targets]
            dt_coord,dt_confidence = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            dt_confidence = (dt_confidence.reshape(batch_size,-1) > pl_confidence_threshold).astype(int)
            gt_coord = np.array([ele['keypoints_ori'] for ele in targets])
            gt_visible = np.array([ele['visible'] > 0 for ele in targets]).astype(int)
            diag = [math.sqrt(ele['box'][-1] ** 2 + ele['box'][-2] ** 2) for ele in targets]

            gt_nums = np.sum(gt_visible, axis=0)
            gt_total_num += gt_nums
            dt_nums = np.sum(dt_confidence,axis=0)
            dt_confident_num += dt_nums

            recall_num = np.zeros(17)
            accurate_num = np.zeros(17)
            for i in range(17):
                dt_vis = dt_confidence[:,i]
                gt_vis = gt_visible[:,i]
                dt_pos = dt_coord[:,i]
                gt_pos = gt_coord[:,i]
                delta_pos = dt_pos - gt_pos
                for j,(x,y) in enumerate(delta_pos):
                    if gt_vis[j] > 0 and dt_vis[j] > 0:
                        recall_num[i] += 1
                        if math.sqrt(x ** 2 + y ** 2) / diag[j] <= pck_threshold:
                            accurate_num[i] += 1

            dt_recall_num += recall_num
            dt_confident_acc_num += accurate_num

    recall = [round(a / b,3) for a,b in zip(dt_recall_num,gt_total_num)]
    precision = [round(a / b,3) for a,b in zip(dt_confident_acc_num,dt_confident_num)]

    kps = animal_kps_info['keypoints']
    for i in range(17):
        # print("Keypoint : {} \t\t\t\t Recall : {} \t\t Precision : {}".format(kps[i],recall[i],precision[i]))
        print("Keypoint : {:<15} Num: {:<4}\t Recall : {:<5.3f}\t Precision : {:<5.3f}".format(kps[i],int(gt_total_num[i]), recall[i], precision[i]))

    # 按照Num对Keypoints进行排序
    sorted_data = sorted(zip(kps, recall, precision, gt_total_num), key=lambda x: x[3],reverse=True)
    sorted_kps, sorted_recall, sorted_precision, sorted_num = zip(*sorted_data)
    ind = np.arange(len(sorted_kps))

    # 绘制Recall和Precision的直方图
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, sorted_recall, 0.35, color='blue', alpha=0.7,label='Recall')
    rects2 = ax.bar(ind + 0.35, sorted_precision, 0.35, color='blue', alpha=0.3,label='Precision')

    # 设置图例和标题
    ax.set_xticks(ind + 0.35 / 2)
    ax.set_xticklabels(sorted_kps)
    plt.xticks(rotation=90)
    plt.legend(loc='best')
    # ax.set_ylabel('Value')
    ax.set_title('Recall and Precision by Keypoint')
    plt.show()

    return


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--name', default="eval", type=str, help='experiment name')
    parser.add_argument('--info', default="", type=str, help='experiment info')
    parser.add_argument('--gpus', default=[0,1], help='device')
    parser.add_argument('--data-root', default='../dataset/ap_10k', type=str, help='data path')
    parser.add_argument('--pretrained-model-path', default='./pretrained_weights',
                        type=str, help='pretrained weights base path')
    parser.add_argument('--pretrained-weights-name', default='ap_10k_SL_best.pth',
                        type=str, help='pretrained weights name')
    parser.add_argument('--output-dir', default='./experiment',type=str, help='output dir depends on the time')

    parser.add_argument('--workers', default=8, type=int, help='number of workers for DataLoader')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size of label data')
    parser.add_argument('--seed', default=2, type=int, help='seed for initializing training')
    # animal body关键点信息
    parser.add_argument('--keypoints-path', default="./info/ap_10k_keypoints_format.json", type=str,
                        help='keypoints_format.json path')
    parser.add_argument('--fixed-size', default=[256, 256], nargs='+', type=int, help='input size')
    # keypoints点数
    parser.add_argument('--num-joints', default=17, type=int, help='num_joints')
    # best info
    parser.add_argument("--amp",default=True,action="store_true",
                        help="Use torch.cuda.amp for mixed precision training")
    # for ScarceNet Test
    parser.add_argument('--cfg',default='outer_tools/experiments/ap10k/hrnet/w32_256x192_adam_lr1e-3_ap10k.yaml',
                        help='experiment configure file name',type=str)
    args = parser.parse_args()

    now = datetime.datetime.now()
    now_time = now.strftime("%Y-%m_%d_%H-%M-%S")
    args.output_dir = os.path.join(args.output_dir,now_time)
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    info_output_dir = os.path.join(output_dir,'info')
    if not os.path.exists(info_output_dir):
        os.mkdir(info_output_dir)
    results_output_dir = os.path.join(output_dir,'results')
    if not os.path.exists(results_output_dir):
        os.mkdir(results_output_dir)
    save_weights_output_dir = os.path.join(output_dir,'save_weights')
    if not os.path.exists(save_weights_output_dir):
        os.mkdir(save_weights_output_dir)

    gpu_list = args.gpus
    str_list = [str(num) for num in gpu_list]  # 将数字列表转换为字符串列表
    gpus = ",".join(str_list)
    os.environ["CUDA_VISIBLE_DEVICE"] = gpus

    update_config(cfg, args)
    logger, final_output_dir = create_logger(
        cfg, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    main(cfg,args)
