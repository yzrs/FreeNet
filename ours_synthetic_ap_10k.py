import datetime
import logging
import argparse
import os
import random
import json
import numpy as np
import torch
import pprint
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader,SubsetRandomSampler
from train_utils.utils import get_cosine_schedule_with_warmup
from models.hrnet import HighResolutionNet
from train_utils import transforms
from train_utils.dataset import CocoKeypoint
from train_utils.ssl_utils import ours_ap10k
from outer_tools.lib.config import cfg,update_config
from outer_tools.lib.utils.utils import create_logger
from torch.backends import cudnn as cudnn


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
    heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)
    kps_weights = np.array(animal_kps_info["kps_weights"],
                           dtype=np.float32).reshape((args.num_joints,))

    data_transform = {
        "train": transforms.Compose([
            transforms.TransformMPL(args, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    data_root = args.data_root
    train_label_dataset = CocoKeypoint(root=data_root, dataset="ap_10k",mode="train",transform=data_transform["train"],
                                       fixed_size=args.fixed_size, data_type="keypoints")
    train_unlabel_dataset = CocoKeypoint(root=data_root, dataset="ap_10k",mode="train",transform=data_transform["train"],
                                         fixed_size=args.fixed_size, data_type="keypoints")

    label_img_id_path = "info/label_list/annotation_list_25.json"
    with open(label_img_id_path, 'r') as f:
        img_ids = json.load(f)

    train_label_dataset.valid_list = [ann for ann in train_label_dataset.valid_list if ann['image_id'] in img_ids]
    train_label_dataset.load_missing_anns(args.anns_info_path)
    train_label_dataset.get_kps_num(args)
    train_unlabel_dataset.get_kps_num(args)

    batch_size = args.batch_size
    nw = args.workers  # number of workers
    logger.info('Using %g dataloader workers' % nw)
    #
    base_weight_path = args.pretrained_model_path
    tea_weight_name = args.pretrained_weights_name
    stu_weight_name = 'pretrained_ori.pth'
    tea_pretrained_weights_path = os.path.join(base_weight_path,tea_weight_name)
    stu_pretrained_weights_path = os.path.join(base_weight_path,stu_weight_name)
    t_model = HighResolutionNet(num_joints=args.num_joints)
    stu_checkpoint = torch.load(stu_pretrained_weights_path)
    tea_checkpoint = torch.load(tea_pretrained_weights_path)

    attr_flag = False
    for key in ['state_dict','model','student_model','directly']:
        if key in tea_checkpoint:
            t_model.load_state_dict(tea_checkpoint[key])
            attr_flag = True
            break
    if not attr_flag:
        t_model.load_state_dict(tea_checkpoint)

    s_model = HighResolutionNet(num_joints=args.num_joints)
    s_model.load_state_dict(stu_checkpoint,strict=False)

    logger.info(f"teacher model loaded from {tea_pretrained_weights_path}:{key}")
    logger.info(f"student model loaded from {stu_pretrained_weights_path}")

    t_model = torch.nn.DataParallel(t_model,device_ids=args.gpus).cuda()
    s_model = torch.nn.DataParallel(s_model,device_ids=args.gpus).cuda()
    t_model.train()
    s_model.train()

    train_label_loader = DataLoader(train_label_dataset,
                                    batch_size=batch_size,
                                    sampler=SubsetRandomSampler(range(len(train_label_dataset))),
                                    pin_memory=True,
                                    num_workers=nw,
                                    drop_last=False,
                                    collate_fn=train_label_dataset.collate_fn_mpl)
    train_unlabel_loader = DataLoader(train_unlabel_dataset,
                                      batch_size=batch_size * args.mu,
                                      sampler=SubsetRandomSampler(range(len(train_unlabel_dataset))),
                                      pin_memory=True,
                                      num_workers=nw,
                                      drop_last=False,
                                      collate_fn=train_unlabel_dataset.collate_fn_mpl)

    t_params = [p for p in t_model.parameters() if p.requires_grad]
    s_params = [p for p in s_model.parameters() if p.requires_grad]

    t_optimizer = torch.optim.AdamW(t_params,lr=args.teacher_lr,weight_decay=args.weight_decay)
    s_optimizer = torch.optim.AdamW(s_params,lr=args.student_lr,weight_decay=args.weight_decay)

    t_scaler = torch.cuda.amp.GradScaler() if args.amp else None
    s_scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps)
    s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps,
                                                  args.student_wait_steps)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_step = checkpoint['step'] + 1
            t_model.module.load_state_dict(checkpoint['teacher_model'])
            s_model.module.load_state_dict(checkpoint['student_model'])
            t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
            s_optimizer.load_state_dict(checkpoint['student_optimizer'])
            t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
            s_scheduler.load_state_dict(checkpoint['student_scheduler'])
            if args.amp:
                t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                s_scaler.load_state_dict(checkpoint['student_scaler'])
            logger.info("=> loaded checkpoint {} (epoch {})".format(args.resume,checkpoint['step']))
        else:
            logger.info("=> no checkpoint found at {}".format(args.resume))

    writer_dir = os.path.join(args.output_dir,"summary")
    if not os.path.exists(writer_dir):
        os.mkdir(writer_dir)
    writer_dict = {
        'writer': SummaryWriter(log_dir=writer_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    ours_ap10k(cfg,args,train_label_loader,train_unlabel_loader,t_model, s_model,t_optimizer,s_optimizer,
               t_scheduler, s_scheduler,t_scaler,s_scaler,writer_dict)

    writer_dict['writer'].close()

    logger.info("Best OKS:{} at Epoch{}".format(
        args.best_oks,args.best_oks_epoch
    ))
    logger.info("Best PCK:{} at Epoch{}".format(
        args.best_pck, args.best_pck_epoch
    ))

    old_name = args.output_dir
    new_name = f"./experiment/{args.name}_{args.info}_{os.path.basename(old_name)}"
    os.rename(old_name,new_name)

    return


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--name', default="ours_for_25_5_imgs_settings", type=str, help='experiment name')
    parser.add_argument('--info', default="", type=str, help='experiment info')
    parser.add_argument('--gpus', default=[0,1], help='device')
    parser.add_argument('--data-root', default='../dataset/ap_10k', type=str, help='data path')
    parser.add_argument('--pretrained-model-path', default='./pretrained_weights',
                        type=str, help='pretrained weights base path')
    parser.add_argument('--pretrained-weights-name', default='25_5_imgs_SL_hrnet_pretrained.pth',
                        type=str, help='pretrained weights name')

    parser.add_argument('--anns-info-path', default='info/25_5_imgs_keypoints_anns_info.json',
                        type=str, help='missing anns info path')
    parser.add_argument('--output-dir', default='./experiment',type=str, help='output dir depends on the time')
    parser.add_argument('--resume', default=None, type=str, help='path to resume file')

    parser.add_argument('--total-steps', default=120000, type=int, help='number of total steps to run')
    parser.add_argument('--eval-step', default=5, type=int, help='number of eval steps to run')
    parser.add_argument('--start-step', default=0, type=int,help='manual epoch number (useful on restarts)')
    parser.add_argument('--warmup-steps', default=900, type=int, help='warmup steps')
    parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')

    parser.add_argument('--uda-steps', default=60000, type=int, help='warmup steps of lambda-u')
    parser.add_argument('--down-step', default=9000,type=int,help='warmup steps of conditional PL')

    parser.add_argument('--feedback-steps-start', default=3000, type=float, help='start steps of feedback')
    parser.add_argument('--feedback-steps-complete', default=6000, type=float, help='warmup steps of feedback')
    parser.add_argument('--feedback-weight', default=2, type=float, help='feedback scalar')

    # 学习率
    parser.add_argument('--teacher_lr', default=1e-5, type=float,
                        help='initial learning rate, 1e-5 is the default value for training')
    parser.add_argument('--student_lr', default=1e-3, type=float,
                        help='initial learning rate, 1e-3 is the default value for training')

    # AdamW的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,metavar='W',
                        help='weight decay (default: 1e-4)',dest='weight_decay')
    parser.add_argument('--grad-clip', default=1e9, type=float, help='gradient norm clipping')
    #
    parser.add_argument('--workers', default=8, type=int, help='number of workers for DataLoader')
    parser.add_argument('--batch-size', default=4, type=int, help='batch size of label data')
    parser.add_argument('--mu', default=1, type=int, help='batch size factor of unlabel data ')
    parser.add_argument('--seed', default=2, type=int, help='seed for initializing training')
    # animal body关键点信息
    parser.add_argument('--keypoints-path', default="./info/ap_10k_keypoints_format.json", type=str,
                        help='keypoints_format.json path')
    parser.add_argument('--fixed-size', default=[256, 256], nargs='+', type=int, help='input size')
    # keypoints点数
    parser.add_argument('--num-joints', default=17, type=int, help='num_joints')
    # best info
    parser.add_argument('--best-oks', default=0, type=float,help='best OKS performance during training')
    parser.add_argument('--best-oks-epoch', default=0, type=int,help='best OKS performance Epoch during training')
    parser.add_argument('--best-pck', default=0, type=float,help='best PCK performance during training')
    parser.add_argument('--best-pck-epoch', default=0, type=int,help='best PCK performance Epoch during training')

    parser.add_argument("--amp",default=True,action="store_true",
                        help="Use torch.cuda.amp for mixed precision training")
    # for ScarceNet Test
    parser.add_argument('--cfg',default='./outer_tools/experiments/ap10k/hrnet/w32_256x192_adam_lr1e-3_ap10k.yaml',
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
