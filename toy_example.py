import datetime
import argparse
import os
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader,SubsetRandomSampler
from train_utils.utils import models_clean,get_cosine_schedule_with_warmup,train_one_epoch
from models.hrnet import create_model, create_model_path
from train_utils import transforms
from train_utils.dataset import MixKeypoint
from train_utils.ssl_utils import train_loop_ssl_v2, train_loop_ssl_self_v2, \
    train_loop_ssl_self_feedback_v2, train_loop_ssl_feedback, train_loop_ssl_split_feedback, train_loop_ssl_uda, \
    train_loop_ssl_self_uda, train_loop_ssl_uda_split_feedback
from train_utils.validation_mix import eval_group_pck,evaluate


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main(args):
    if args.seed is not None:
        set_seed(args)
    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    fixed_size = args.fixed_size
    heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)
    kps_weights = np.array(animal_kps_info["kps_weights"],dtype=np.float32).reshape((args.num_joints,))

    data_transform = {
        "train_sl": transforms.Compose([
            transforms.LabelFormatTrans(extend_flag=True),
            transforms.HalfBody(0.3, animal_kps_info["upper_body_ids"], animal_kps_info["lower_body_ids"]),
            transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            transforms.RandomHorizontalFlip(0.5, animal_kps_info["flip_pairs"]),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            # RandWeakAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "train": transforms.Compose([
            transforms.LabelFormatTrans(extend_flag=True),
            transforms.TransformMPL(args, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.LabelFormatTrans(extend_flag=True),
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    train_dataset_info = [{"dataset":"ap_10k","mode":"train"},{"dataset":"animal_pose","mode":"train"}]
    val_dataset_info = [{"dataset":"ap_10k","mode":"val"},{"dataset":"animal_pose","mode":"val"}]
    data_root = args.data_root

    if args.sl:
        train_dataset = MixKeypoint(root=data_root, merge_info=train_dataset_info, transform=data_transform['train_sl'])
        val_dataset = MixKeypoint(root=data_root, merge_info=val_dataset_info, transform=data_transform['val'])
        train_dataset.sample_few(num_ratio=0.01)
        train_dataset.get_kps_num(args)

        batch_size = args.batch_size
        nw = args.workers  # number of workers
        print('Using %g dataloader workers' % nw)

        base_weight_path = args.pretrained_model_path
        model = create_model(num_joints=args.num_joints, weight_path=base_weight_path, load_pretrain_weights=True)
        model.to(args.device)
        model.train()
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  sampler=SubsetRandomSampler(range(len(train_dataset))),
                                  pin_memory=True,
                                  num_workers=nw,
                                  drop_last=False,
                                  collate_fn=train_dataset.collate_fn)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params,
                                      lr=args.lr,
                                      weight_decay=args.weight_decay
                                      )
        scaler = torch.cuda.amp.GradScaler() if args.amp else None
        # learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
        val_path = os.path.join(args.output_dir, "info/val_log.txt")

        for epoch in range(args.start_epoch, args.epochs):
            model.train()
            mean_loss, lr = train_one_epoch(args, model, optimizer, train_loader, device=args.device, epoch=epoch,
                                            warmup=True, scaler=scaler)
            lr_scheduler.step()

            if (epoch + 1) % args.eval_epoch == 0:
                model_name = f"model-{epoch}"
                model.eval()
                # save weights
                save_files = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch}
                if args.amp:
                    save_files["scaler"] = scaler.state_dict()
                torch.save(save_files, "./{}/save_weights/model-{}.pth".format(args.output_dir, epoch))
                # evaluate on the test dataset
                val_oks_value, val_pck_value, oks_list, pck_list = evaluate(args=args, model_name=model_name,
                                                                            dataset=val_dataset)
                oks_dict = {key['dataset'] + '_' + key['mode']: val for key, val in
                            zip(val_dataset.dataset_infos, oks_list)}
                pck_dict = {key['dataset'] + '_' + key['mode']: val for key, val in
                            zip(val_dataset.dataset_infos, pck_list)}

                # 计算在shared keypoints 和 exclusive keypoint上的PCK
                # 这里是AP-10K 和 Animal Pose
                # shared keypoints: [0,1,4,11,12,13,14,15,16,17,18,21,22,23,24,25]
                # exclusive keypoints:[[8],[2,3,6,7]]
                shared_kp_index = [0, 1, 4, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25]
                exclusive_kp_index_a = [8]
                exclusive_kp_index_b = [2, 3, 6, 7]
                avg_shared_kps_pck, _ = eval_group_pck(args, model_name, val_dataset, shared_kp_index, [0, 1])
                ap_10k_exclusive_kps_pck, _ = eval_group_pck(args, model_name, val_dataset, exclusive_kp_index_a, [0])
                animal_pose_exclusive_kps_pck, _ = eval_group_pck(args, model_name, val_dataset, exclusive_kp_index_b,
                                                                  [1])
                # write into txt
                with open(val_path, "a") as f:
                    # 写入的数据包括coco指标还有loss和learning rate
                    result_info = [
                        f"val_mean_oks:{val_oks_value}",
                        f"val_mean_pck:{val_pck_value}",
                        f"loss:{mean_loss}:.6f",
                        f"learning_rate:{lr:.6f}",
                        f"oks_dict: {' '.join([f'{k}: {v}' for k, v in oks_dict.items()])}",
                        f"pck_dict: {' '.join([f'{k}: {v}' for k, v in pck_dict.items()])}",
                        f"PCK on shared kps:{avg_shared_kps_pck:.4f}",
                        f"PCK on exclusive kps a:{ap_10k_exclusive_kps_pck:.4f}",
                        f"PCK on exclusive kps b:{animal_pose_exclusive_kps_pck:.4f}"
                    ]
                    txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                    f.write(txt + "\n")
    else:
        train_label_dataset = MixKeypoint(root=data_root, merge_info=train_dataset_info,transform=data_transform['train'])
        train_unlabel_dataset = MixKeypoint(root=data_root, merge_info=train_dataset_info,transform=data_transform['train'])
        val_dataset = MixKeypoint(root=data_root, merge_info=val_dataset_info, transform=data_transform['val'])

        train_label_dataset.sample_few(num_ratio=0.01)
        train_unlabel_dataset.eliminate_repeated_data(train_label_dataset.valid_lists)
        train_label_dataset.get_kps_num(args)
        train_unlabel_dataset.get_kps_num(args)
        print("Data num of label dataset: ",len(train_label_dataset))
        print("Data num of unlabel dataset: ",len(train_unlabel_dataset))

        batch_size = args.batch_size
        nw = args.workers  # number of workers
        print('Using %g dataloader workers' % nw)
        #
        base_weight_path = args.pretrained_model_path
        t_model = create_model_path(num_joints=args.num_joints, weight_path=base_weight_path,
                                    weight_name="mix_SL_0.1.pth")
        t_model.to(args.device)
        t_model.train()

        s_model = create_model(num_joints=args.num_joints, weight_path=base_weight_path, load_pretrain_weights=True)
        s_model.to(args.device)
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

        t_optimizer = torch.optim.AdamW(t_params, lr=args.teacher_lr, weight_decay=args.weight_decay)
        s_optimizer = torch.optim.AdamW(s_params, lr=args.student_lr, weight_decay=args.weight_decay)

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

        # train
        if args.ssl:
            train_loop_ssl_v2(args, train_label_loader, train_unlabel_loader, t_model, s_model, t_optimizer,
                              s_optimizer,t_scheduler, s_scheduler, t_scaler, s_scaler, val_dataset)
        elif args.ssl_self:
            train_loop_ssl_self_v2(args, train_label_loader, train_unlabel_loader, t_model, s_model, t_optimizer,
                                   s_optimizer,t_scheduler, s_scheduler, t_scaler, s_scaler, val_dataset)
        elif args.ssl_uda:
            train_loop_ssl_uda(args, train_label_loader, train_unlabel_loader, t_model, s_model, t_optimizer,
                               s_optimizer,t_scheduler, s_scheduler, t_scaler, s_scaler, val_dataset)
        elif args.ssl_feedback:
            train_loop_ssl_feedback(args, train_label_loader, train_unlabel_loader, t_model, s_model, t_optimizer,
                                    s_optimizer,t_scheduler, s_scheduler, t_scaler, s_scaler, val_dataset)
        elif args.ssl_self_feedback:
            train_loop_ssl_self_feedback_v2(args, train_label_loader, train_unlabel_loader, t_model, s_model, t_optimizer,
                                            s_optimizer,t_scheduler, s_scheduler, t_scaler, s_scaler, val_dataset)
        elif args.ssl_self_uda:
            train_loop_ssl_self_uda(args, train_label_loader, train_unlabel_loader, t_model, s_model, t_optimizer,
                                    s_optimizer,t_scheduler, s_scheduler, t_scaler, s_scaler, val_dataset)
        elif args.ssl_split_feedback:
            train_loop_ssl_split_feedback(args, train_label_loader, train_unlabel_loader, t_model, s_model, t_optimizer,
                                          s_optimizer,t_scheduler, s_scheduler, t_scaler, s_scaler, val_dataset)
        elif args.ssl_uda_split_feedback:
            train_loop_ssl_uda_split_feedback(args, train_label_loader, train_unlabel_loader, t_model, s_model,t_optimizer,
                                              s_optimizer,t_scheduler, s_scheduler, t_scaler, s_scaler, val_dataset)
        else:
            return

    if not args.save_all_weights:
        models_clean(root_path=args.output_dir,save_num=5,has_oks=True,has_pck=True)
    old_name = args.output_dir
    new_name = f"./experiment/{args.name}_{args.info}_{os.path.basename(old_name)}"
    os.rename(old_name,new_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--name', default='mix', type=str, help='experiment name')
    parser.add_argument('--info', default="toy", type=str, help='experiment info')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--data-root', default='../dataset', type=str, help='data path')
    parser.add_argument('--pretrained-model-path', default='./pretrained_weights',type=str, help='pretrained weights path')
    parser.add_argument('--output-dir', default='./experiment',type=str, help='output dir depends on the time')

    # SL params
    parser.add_argument('--lr', default=5e-4, type=float,help='initial learning rate, 5e-4 is the default value for training')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--eval-epoch', default=1, type=int, help='train x epoch, evaluate once')
    parser.add_argument('--lr-steps', default=[5, 8], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')

    # SSL params
    parser.add_argument('--total-steps', default=100, type=int, help='number of total steps to run')
    parser.add_argument('--eval-step', default=10, type=int, help='number of eval steps to run')
    parser.add_argument('--start-step', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--warmup-steps', default=30, type=int, help='warmup steps')
    parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
    parser.add_argument('--uda-steps', default=50, type=float, help='warmup steps of lambda-u')

    # SSL params
    parser.add_argument('--teacher_lr', default=1e-5, type=float,help='initial learning rate, 1e-5 is the default value for training')
    parser.add_argument('--student_lr', default=1e-3, type=float,help='initial learning rate, 1e-3 is the default value for training')

    # AdamW的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,metavar='W',
                        help='weight decay (default: 1e-4)',dest='weight_decay')
    parser.add_argument('--grad-clip', default=1e9, type=float, help='gradient norm clipping')
    #
    parser.add_argument('--workers', default=8, type=int, help='number of workers for DataLoader')
    parser.add_argument('--batch-size', default=8, type=int, help='batch size of label data')
    parser.add_argument('--mu', default=1, type=int, help='batch size factor of unlabel data ')
    parser.add_argument('--seed', default=2, type=int, help='seed for initializing training')
    # animal body关键点信息
    parser.add_argument('--keypoints-path', default="./info/keypoints_definition.json", type=str,
                        help='keypoints_format.json path')
    parser.add_argument('--fixed-size', default=[256, 256], nargs='+', type=int, help='input size')
    # keypoints点数
    parser.add_argument('--num-joints', default=26, type=int, help='num_joints')
    parser.add_argument("--all_results", action="store_true",  help="if true, evaluation will also save 26 kps prediction")
    parser.add_argument("--save-all-weights", action="store_true",  help="whether save weights of all the epochs")
    parser.add_argument("--finetune", action="store_true",  help="whether student model use pretrained "
                                                                 "weights same as teacher ")

    parser.add_argument("--sl",action="store_true",help="experiment setting")
    parser.add_argument("--ssl",default=True,action="store_true",help="experiment setting")
    parser.add_argument("--ssl_self", action="store_true",help="experiment setting")
    parser.add_argument("--ssl_uda", action="store_true",help="experiment setting")
    parser.add_argument("--ssl_feedback", action="store_true",help="experiment setting")
    parser.add_argument("--ssl_self_feedback", action="store_true",help="experiment setting")
    parser.add_argument("--ssl_self_uda", action="store_true",help="experiment setting")
    parser.add_argument("--ssl_split_feedback", action="store_true",help="experiment setting")
    parser.add_argument("--ssl_uda_split_feedback", action="store_true",help="experiment setting")

    parser.add_argument("--feedback_shared", default=False,action="store_true",help="feedback loss for mixing training")
    parser.add_argument("--feedback_ap_10k_exclusive", action="store_true",help="feedback loss for mixing training")
    parser.add_argument("--feedback_animal_pose_exclusive", action="store_true",help="feedback loss for mixing training")

    parser.add_argument("--amp",default=True,action="store_true",  help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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

    if args.workers > 0:
        args.workers = min([os.cpu_count(), args.workers])
    else:
        args.workers = os.cpu_count()
    main(args)