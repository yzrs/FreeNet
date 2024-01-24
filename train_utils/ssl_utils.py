import json
import os
import time
from torch import nn
from torch.cuda import amp
from tqdm import tqdm
from train_utils.transforms import get_max_preds
from train_utils.utils import (AverageMeter, generate_heatmap,
                               AvgImgMSELoss, get_current_topkrate)
from train_utils import transforms
from train_utils.validation_mix import eval_group_pck, eval_model_parallel
import torch
from torch.utils.data import DataLoader
from outer_tools.lib.core.loss import JointsMSELoss
from outer_tools.lib.core.function import validate
from outer_tools.lib import dataset_animal
from torchvision import transforms as tf
import logging
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def ours_ap10k_consistency(cfg, args, labeled_loader, unlabeled_loader, teacher_model, student_model,t_optimizer,s_optimizer,
                           t_scheduler, s_scheduler, t_scaler, s_scaler, writer_dict):
    """
    :param writer_dict: writer
    :param args: ..
    :param labeled_loader: ..
    :param unlabeled_loader: ..
    :param teacher_model: ..
    :param student_model: ..
    :param t_optimizer: ..
    :param s_optimizer: ..
    :param t_scheduler: ..
    :param s_scheduler: ..
    :param t_scaler: ..
    :param s_scaler: ..
    :return: none
    """
    # update args info
    program_info_path = os.path.join(args.output_dir, "info", "program_info.txt")
    args.info = "conditional_PL_conditional_feedback"
    args_str = json.dumps(vars(args))
    with open(program_info_path, "w") as f:
        f.write(args_str)
    logger.info(args_str)

    with open(args.keypoints_path, 'r') as f:
        kps_info = json.load(f)
    kps_weights = kps_info['kps_weights']
    criterion = AvgImgMSELoss(kps_weights=kps_weights, num_joints=args.num_joints)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    teacher_model.train()
    student_model.train()

    s_optimizer.zero_grad()
    t_optimizer.zero_grad()

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()

        end = time.time()

        # train
        try:
            (images_l_weak, images_l_strong), (weak_label_targets,strong_label_targets) = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            (images_l_weak, images_l_strong), (weak_label_targets,strong_label_targets) = next(labeled_iter)
        except Exception as e:
            # 处理其他特定异常情况
            # 可以输出异常信息等
            logger.error("An error occurred:", e)
            return
        try:
            (images_u_weak, images_u_strong), (weak_unlabel_targets,strong_unlabel_targets) = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            (images_u_weak, images_u_strong), (weak_unlabel_targets,strong_unlabel_targets) = next(unlabeled_iter)
        except Exception as e:
            logger.error("An error occurred:", e)
            return

        data_time.update(time.time() - end)

        images_l_weak = images_l_weak.cuda()
        images_l_strong = images_l_strong.cuda()
        images_u_weak = images_u_weak.cuda()
        images_u_strong = images_u_strong.cuda()

        with amp.autocast(enabled=args.amp):
            label_batch_size = images_l_weak.shape[0]
            images_weak = torch.cat((images_l_weak, images_u_weak)).contiguous()
            images_strong = torch.cat((images_l_strong, images_u_strong)).contiguous()
            # 使teacher model 和 student model 的batch normalization层相似
            with torch.no_grad():
                _ = student_model(images_weak)

            t_logits_strong = teacher_model(images_strong)
            s_logits_weak = student_model(images_strong)
            t_logits_weak = teacher_model(images_weak)

            t_logits_l = t_logits_weak[:label_batch_size]
            t_logits_u = t_logits_weak[label_batch_size:]
            s_logits_l = s_logits_weak[:label_batch_size]
            s_logits_u = s_logits_weak[label_batch_size:]
            t_logits_aug_u = t_logits_strong[label_batch_size:]

            # semi-supervise 硬伪标签
            weak_coords, weak_confidence = get_max_preds(t_logits_u)
            strong_coords, strong_confidence = get_max_preds(t_logits_aug_u)

            weak_confidence = weak_confidence.float().squeeze(-1)
            strong_confidence = strong_confidence.float().squeeze(-1)

            weak_reverse_trans = torch.tensor([t['reverse_trans'] for t in weak_label_targets]).to(torch.float16).cuda()
            strong_trans = torch.tensor([t['trans'] for t in strong_label_targets]).to(torch.float16).cuda()
            # [64,64] -> [256,256]
            weak_trans_coords = weak_coords * 4
            for i in range(t_logits_aug_u.shape[0]):
                # reverse to original coord
                mask = torch.logical_and(weak_trans_coords[i,:,0] > 0, weak_trans_coords[i,:,1] > 0)
                pt = weak_trans_coords[i,mask]
                t = weak_reverse_trans[i]
                ones = torch.ones((pt.shape[0],1),dtype=torch.float16).cuda()
                pt = torch.cat((pt,ones),dim=1).T
                weak_trans_coords[i,mask] = torch.mm(t,pt).T

                pt = weak_trans_coords[i, mask]
                t = strong_trans[i]
                ones = torch.ones((pt.shape[0], 1), dtype=torch.float16).cuda()
                pt = torch.cat((pt, ones), dim=1).T
                weak_trans_coords[i,mask] = torch.mm(t,pt).T

            print('debug')

        #     target_heatmaps = torch.stack([t["heatmap"] for t in weak_label_targets]).cuda(non_blocking=True)
        #     target_visible = torch.stack([torch.tensor(t["visible"]) for t in weak_label_targets])
        #     target_visible[target_visible != 0] = 1
        #
        #     # Conditional PL
        #     group_indice_face = [0, 1, 2]
        #     group_indice_front = [5, 6, 7, 8, 9, 10]
        #     group_indice_back = [4, 11, 12, 13, 14, 15, 16]
        #     group_indice_exclusive = [3]
        #     group_indices = [group_indice_face, group_indice_front, group_indice_back,group_indice_exclusive]
        #     confidence_thresholds = []
        #     # random down
        #     # min_ratios = [0.9, 0.75, 0.6, 0.5]
        #     min_ratios = [0.5, 0.6, 0.65, 0.65]
        #     # 扫描其中所有confidence 并排序
        #     for i, indices in enumerate(group_indices):
        #         group_confidences = []
        #         for index in indices:
        #             for j in range(tea_u_confidence.shape[0]):
        #                 group_confidences.append(tea_u_confidence[j][index].item())
        #         group_confidences = sorted(group_confidences, reverse=True)
        #         cur_ratio = get_current_topkrate(step, args.down_step, min_rate=min_ratios[i])
        #         sample_nums = len(group_confidences)
        #         cur_confidence_th = max(group_confidences[min(int(sample_nums * cur_ratio), sample_nums - 1)], 0.2)
        #         confidence_thresholds.append(cur_confidence_th)
        #
        #     tea_pseudo_visible = torch.zeros_like(tea_u_confidence)
        #     for i, indices in enumerate(group_indices):
        #         for index in indices:
        #             tea_pseudo_visible[:, index] = tea_u_confidence[:, index] >= confidence_thresholds[i]
        #     tea_pseudo_labels = generate_heatmap(coords, tea_pseudo_visible.cpu()).cuda()
        #
        #     s_loss_l = criterion(s_logits_l, target_heatmaps, target_visible)
        #     s_loss_pl = criterion(s_logits_u, tea_pseudo_labels, tea_pseudo_visible)
        #     s_loss = s_loss_l + s_loss_pl
        #
        # s_scaler.scale(s_loss).backward()
        # if args.grad_clip > 0:
        #     s_scaler.unscale_(s_optimizer)
        #     nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        # s_scaler.step(s_optimizer)
        # s_scaler.update()
        # s_scheduler.step()
        #
        # with amp.autocast(enabled=args.amp):
        #     t_loss_l = criterion(t_logits_l, target_heatmaps, target_visible)
        #
        #     # conditional feedback for those sensitive keypoints
        #     if step >= args.feedback_steps_start:
        #         with torch.no_grad():
        #             s_logits_l_new = student_model(images_l_aug)
        #         s_loss_l_new = criterion(s_logits_l_new, target_heatmaps, target_visible)
        #         dot_product = s_loss_l.detach() - s_loss_l_new
        #
        #         ratio_visible_good = 0.2
        #         ratio_invisible_bad = 0.2
        #         tea_u_kps_confidence = torch.clone(tea_u_confidence.detach()).flatten().cpu().numpy()
        #         _, stu_u_confidence = get_max_preds(s_logits_u)
        #         stu_u_kps_confidence = torch.clone(stu_u_confidence.detach()).flatten().cpu().numpy()
        #         tea_u_kps_confidence = sorted(tea_u_kps_confidence, reverse=True)
        #         stu_u_kps_confidence = sorted(stu_u_kps_confidence, reverse=True)
        #         min_good_index = int(len(tea_u_kps_confidence) * ratio_visible_good)
        #         max_bad_index = int(len(stu_u_kps_confidence) * (1 - ratio_invisible_bad))
        #         min_good_th = max(tea_u_kps_confidence[min_good_index], stu_u_kps_confidence[min_good_index])
        #         max_bad_th = min(tea_u_kps_confidence[max_bad_index], stu_u_kps_confidence[max_bad_index])
        #         feedback_vis = torch.clone(tea_u_confidence.detach())
        #         feedback_vis = torch.where((feedback_vis >= max_bad_th) & (feedback_vis <= min_good_th),
        #                                    torch.ones_like(feedback_vis), torch.zeros_like(feedback_vis))
        #         # feedback_vis = torch.where((feedback_vis >= max_bad_th),torch.ones_like(feedback_vis), torch.zeros_like(feedback_vis))
        #         # feedback hard heatmap
        #         coords_aug, _ = get_max_preds(t_logits_aug_u)
        #         t_logits_aug_u_pl = generate_heatmap(coords_aug, feedback_vis.cpu()).cuda()
        #         feedback_term = criterion(t_logits_aug_u, t_logits_aug_u_pl, feedback_vis)
        #         t_loss_feedback = dot_product * feedback_term
        #         feedback_factor = min(1.,(step + 1 - args.feedback_steps_start) /
        #                               (args.feedback_steps_complete - args.feedback_steps_start)) * args.feedback_weight
        #         t_loss = t_loss_l + t_loss_feedback * feedback_factor
        #
        #     else:
        #         t_loss = t_loss_l
        #
        # t_scaler.scale(t_loss).backward()
        # if args.grad_clip > 0:
        #     t_scaler.unscale_(t_optimizer)
        #     nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        # t_scaler.step(t_optimizer)
        # t_scaler.update()
        # t_scheduler.step()
        #
        # s_optimizer.zero_grad()
        # t_optimizer.zero_grad()
        #
        # s_losses.update(s_loss.item())
        # t_losses.update(t_loss.item())
        #
        # batch_time.update(time.time() - end)
        # pbar.set_description(
        #     f"Train Iter: {step:4}/{args.total_steps:4}. "
        #     f"S_LR: {s_optimizer.param_groups[0]['lr']:.5f}. T_LR: {t_optimizer.param_groups[0]['lr']:.5f}. "
        #     f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. T_Loss: {t_losses.avg:.4f}. ")
        # pbar.update()
        #
        # # evaluate
        # if (step + 1) % args.eval_step == 0:
        #     train_loss = {'Stu_Loss': s_losses.avg, 'Tea_Loss': t_losses.avg}
        #     writer = writer_dict['writer']
        #     global_steps = writer_dict['train_global_steps']
        #     writer.add_scalars('Train_Loss', train_loss, global_steps)
        #     writer_dict['train_global_steps'] = global_steps + 1
        #
        #     pbar.close()
        #     teacher_model.eval()
        #     student_model.eval()
        #     ap10k_eval_model(cfg, args, step, teacher_model, student_model, t_optimizer, s_optimizer,
        #                      t_scheduler,s_scheduler, t_scaler, s_scaler, writer_dict)
        #     teacher_model.train()
        #     student_model.train()


def ours_ap10k(cfg, args, labeled_loader, unlabeled_loader, teacher_model, student_model,t_optimizer,s_optimizer,
               t_scheduler, s_scheduler, t_scaler, s_scaler, writer_dict):
    """
    :param writer_dict: writer
    :param args: ..
    :param labeled_loader: ..
    :param unlabeled_loader: ..
    :param teacher_model: ..
    :param student_model: ..
    :param t_optimizer: ..
    :param s_optimizer: ..
    :param t_scheduler: ..
    :param s_scheduler: ..
    :param t_scaler: ..
    :param s_scaler: ..
    :return: none
    """
    # update args info
    program_info_path = os.path.join(args.output_dir, "info", "program_info.txt")
    args.info = "conditional_PL_conditional_feedback"
    args_str = json.dumps(vars(args))
    with open(program_info_path, "w") as f:
        f.write(args_str)
    logger.info(args_str)

    with open(args.keypoints_path, 'r') as f:
        kps_info = json.load(f)
    kps_weights = kps_info['kps_weights']
    criterion = AvgImgMSELoss(kps_weights=kps_weights, num_joints=args.num_joints)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    teacher_model.train()
    student_model.train()

    s_optimizer.zero_grad()
    t_optimizer.zero_grad()

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()

        end = time.time()

        # train
        try:
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except Exception as e:
            # 处理其他特定异常情况
            # 可以输出异常信息等
            logger.error("An error occurred:", e)
            return
        try:
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except Exception as e:
            logger.error("An error occurred:", e)
            return

        data_time.update(time.time() - end)

        images_l_ori = images_l_ori.cuda()
        images_l_aug = images_l_aug.cuda()
        images_u_ori = images_u_ori.cuda()
        images_u_aug = images_u_aug.cuda()

        with amp.autocast(enabled=args.amp):
            label_batch_size = images_l_ori.shape[0]
            images_ori = torch.cat((images_l_ori, images_u_ori)).contiguous()
            images_aug = torch.cat((images_l_aug, images_u_aug)).contiguous()
            # 使teacher model 和 student model 的batch normalization层相似
            with torch.no_grad():
                _ = student_model(images_ori)

            t_logits_aug = teacher_model(images_aug)
            s_logits_aug = student_model(images_aug)
            t_logits_ori = teacher_model(images_ori)

            t_logits_l = t_logits_ori[:label_batch_size]
            t_logits_u = t_logits_ori[label_batch_size:]
            s_logits_l = s_logits_aug[:label_batch_size]
            s_logits_u = s_logits_aug[label_batch_size:]
            t_logits_aug_u = t_logits_aug[label_batch_size:]

            target_heatmaps = torch.stack([t["heatmap"] for t in targets]).cuda(non_blocking=True)
            target_visible = torch.stack([torch.tensor(t["visible"]) for t in targets])
            target_visible[target_visible != 0] = 1

            # semi-supervise 硬伪标签
            coords, tea_u_confidence = get_max_preds(t_logits_u)
            tea_u_confidence = tea_u_confidence.float().squeeze(-1)

            # Conditional PL
            group_indice_face = [0, 1, 2]
            group_indice_front = [5, 6, 7, 8, 9, 10]
            group_indice_back = [4, 11, 12, 13, 14, 15, 16]
            group_indice_exclusive = [3]
            group_indices = [group_indice_face, group_indice_front, group_indice_back,group_indice_exclusive]
            confidence_thresholds = []
            # random down
            # min_ratios = [0.9, 0.75, 0.6, 0.5]
            min_ratios = [0.5, 0.6, 0.65, 0.65]
            # 扫描其中所有confidence 并排序
            for i, indices in enumerate(group_indices):
                group_confidences = []
                for index in indices:
                    for j in range(tea_u_confidence.shape[0]):
                        group_confidences.append(tea_u_confidence[j][index].item())
                group_confidences = sorted(group_confidences, reverse=True)
                cur_ratio = get_current_topkrate(step, args.down_step, min_rate=min_ratios[i])
                sample_nums = len(group_confidences)
                cur_confidence_th = max(group_confidences[min(int(sample_nums * cur_ratio), sample_nums - 1)], 0.2)
                confidence_thresholds.append(cur_confidence_th)

            tea_pseudo_visible = torch.zeros_like(tea_u_confidence)
            for i, indices in enumerate(group_indices):
                for index in indices:
                    tea_pseudo_visible[:, index] = tea_u_confidence[:, index] >= confidence_thresholds[i]
            tea_pseudo_labels = generate_heatmap(coords, tea_pseudo_visible.cpu()).cuda()

            s_loss_l = criterion(s_logits_l, target_heatmaps, target_visible)
            s_loss_pl = criterion(s_logits_u, tea_pseudo_labels, tea_pseudo_visible)
            s_loss = s_loss_l + s_loss_pl

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()

        with amp.autocast(enabled=args.amp):
            t_loss_l = criterion(t_logits_l, target_heatmaps, target_visible)

            # conditional feedback for those sensitive keypoints
            if step >= args.feedback_steps_start:
                with torch.no_grad():
                    s_logits_l_new = student_model(images_l_aug)
                s_loss_l_new = criterion(s_logits_l_new, target_heatmaps, target_visible)
                dot_product = s_loss_l.detach() - s_loss_l_new

                ratio_visible_good = 0.2
                ratio_invisible_bad = 0.2
                tea_u_kps_confidence = torch.clone(tea_u_confidence.detach()).flatten().cpu().numpy()
                _, stu_u_confidence = get_max_preds(s_logits_u)
                stu_u_kps_confidence = torch.clone(stu_u_confidence.detach()).flatten().cpu().numpy()
                tea_u_kps_confidence = sorted(tea_u_kps_confidence, reverse=True)
                stu_u_kps_confidence = sorted(stu_u_kps_confidence, reverse=True)
                min_good_index = int(len(tea_u_kps_confidence) * ratio_visible_good)
                max_bad_index = int(len(stu_u_kps_confidence) * (1 - ratio_invisible_bad))
                min_good_th = max(tea_u_kps_confidence[min_good_index], stu_u_kps_confidence[min_good_index])
                max_bad_th = min(tea_u_kps_confidence[max_bad_index], stu_u_kps_confidence[max_bad_index])
                feedback_vis = torch.clone(tea_u_confidence.detach())
                feedback_vis = torch.where((feedback_vis >= max_bad_th) & (feedback_vis <= min_good_th),
                                           torch.ones_like(feedback_vis), torch.zeros_like(feedback_vis))
                # feedback_vis = torch.where((feedback_vis >= max_bad_th),torch.ones_like(feedback_vis), torch.zeros_like(feedback_vis))
                # feedback hard heatmap
                coords_aug, _ = get_max_preds(t_logits_aug_u)
                t_logits_aug_u_pl = generate_heatmap(coords_aug, feedback_vis.cpu()).cuda()
                feedback_term = criterion(t_logits_aug_u, t_logits_aug_u_pl, feedback_vis)
                t_loss_feedback = dot_product * feedback_term
                feedback_factor = min(1.,(step + 1 - args.feedback_steps_start) /
                                      (args.feedback_steps_complete - args.feedback_steps_start)) * args.feedback_weight
                t_loss = t_loss_l + t_loss_feedback * feedback_factor

            else:
                t_loss = t_loss_l

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        s_optimizer.zero_grad()
        t_optimizer.zero_grad()

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step:4}/{args.total_steps:4}. "
            f"S_LR: {s_optimizer.param_groups[0]['lr']:.5f}. T_LR: {t_optimizer.param_groups[0]['lr']:.5f}. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. T_Loss: {t_losses.avg:.4f}. ")
        pbar.update()

        # evaluate
        if (step + 1) % args.eval_step == 0:
            train_loss = {'Stu_Loss': s_losses.avg, 'Tea_Loss': t_losses.avg}
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalars('Train_Loss', train_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            pbar.close()
            teacher_model.eval()
            student_model.eval()
            ap10k_eval_model(cfg, args, step, teacher_model, student_model, t_optimizer, s_optimizer,
                             t_scheduler,s_scheduler, t_scaler, s_scaler, writer_dict)
            teacher_model.train()
            student_model.train()


def ours_ap10k_animalpose_draw(cfg, args, labeled_loader, unlabeled_loader, teacher_model, student_model,t_optimizer,s_optimizer,
                          t_scheduler, s_scheduler, t_scaler, s_scaler, writer_dict):
    """
    :param writer_dict: writer
    :param args: ..
    :param labeled_loader: ..
    :param unlabeled_loader: ..
    :param teacher_model: ..
    :param student_model: ..
    :param t_optimizer: ..
    :param s_optimizer: ..
    :param t_scheduler: ..
    :param s_scheduler: ..
    :param t_scaler: ..
    :param s_scaler: ..
    :return: none
    """
    # update args info
    program_info_path = os.path.join(args.output_dir, "info", "program_info.txt")
    args.info = "conditional_PL_conditional_feedback"
    args_str = json.dumps(vars(args))
    with open(program_info_path, "w") as f:
        f.write(args_str)
    logger.info(args_str)

    with open(args.keypoints_path, 'r') as f:
        kps_info = json.load(f)
    kps_weights = kps_info['kps_weights']
    criterion = AvgImgMSELoss(kps_weights=kps_weights, num_joints=args.num_joints)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    teacher_model.train()
    student_model.train()

    s_optimizer.zero_grad()
    t_optimizer.zero_grad()

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()

        end = time.time()

        # train
        try:
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except Exception as e:
            # 处理其他特定异常情况
            # 可以输出异常信息等
            logger.error("An error occurred:", e)
            return
        try:
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except Exception as e:
            logger.error("An error occurred:", e)
            return

        data_time.update(time.time() - end)

        images_l_ori = images_l_ori.cuda()
        images_l_aug = images_l_aug.cuda()
        images_u_ori = images_u_ori.cuda()
        images_u_aug = images_u_aug.cuda()

        with amp.autocast(enabled=args.amp):
            label_batch_size = images_l_ori.shape[0]
            images_ori = torch.cat((images_l_ori, images_u_ori)).contiguous()
            images_aug = torch.cat((images_l_aug, images_u_aug)).contiguous()
            # 使teacher model 和 student model 的batch normalization层相似
            with torch.no_grad():
                _ = student_model(images_ori)

            t_logits_aug = teacher_model(images_aug)
            s_logits_aug = student_model(images_aug)
            t_logits_ori = teacher_model(images_ori)

            t_logits_l = t_logits_ori[:label_batch_size]
            t_logits_u = t_logits_ori[label_batch_size:]
            s_logits_l = s_logits_aug[:label_batch_size]
            s_logits_u = s_logits_aug[label_batch_size:]
            t_logits_aug_u = t_logits_aug[label_batch_size:]

            target_heatmaps = torch.stack([t["heatmap"] for t in targets]).cuda(non_blocking=True)
            target_visible = torch.stack([torch.tensor(t["visible"]) for t in targets])
            target_visible[target_visible != 0] = 1

            gt_coords,gt_vis = get_max_preds(target_heatmaps)
            gt_coords = gt_coords * 4
            gt_seperate_heatmaps = generate_heatmap(gt_coords,gt_vis,heatmap_size=(256,256))
            gt_heatmaps = torch.sum(gt_seperate_heatmaps,dim=1,keepdim=False).cpu()

            dt_coords,dt_vis = get_max_preds(t_logits_l)
            dt_vis = (dt_vis>0.1).float()
            dt_coords = dt_coords * 4
            dt_seperate_heatmaps = generate_heatmap(dt_coords,dt_vis,heatmap_size=(256,256))
            dt_heatmaps = torch.sum(dt_seperate_heatmaps,dim=1,keepdim=False).cpu()

            draw = True
            if draw:
                for i in range(args.batch_size):
                    # 假设你的图像tensor是images，且Normalize的均值和标准差是mean和std
                    mean = torch.tensor([0.485, 0.456, 0.406])
                    std = torch.tensor([0.229, 0.224, 0.225])

                    # 反归一化
                    images = images_l_ori[i].cpu()  # 去掉批次维度
                    images = images * std.view(3, 1, 1) + mean.view(3, 1, 1)  # 反归一化
                    images = images.clamp(0, 1)  # 将值裁剪到0-1的范围内

                    # 转换为PIL图像
                    to_pil = tf.ToPILImage()
                    image = to_pil(images)

                    fig,ax = plt.subplots(1,2)
                    im0 = ax[0].imshow(image)
                    ax[0].imshow(gt_heatmaps[i], alpha=0.5, cmap='hot')
                    im1 = ax[1].imshow(image)
                    ax[1].imshow(dt_heatmaps[i], alpha=0.5, cmap='hot')
                    plt.show()

            # semi-supervise 硬伪标签
            coords, tea_u_confidence = get_max_preds(t_logits_u)
            tea_u_confidence = tea_u_confidence.float().squeeze(-1)

            # Conditional PL
            group_indice_face = [0, 1, 2]
            # Ear Re-grouping
            group_indice_front = [5, 6, 7, 8, 9, 10, 17, 18]
            group_indice_back = [4, 11, 12, 13, 14, 15, 16]
            group_indice_exclusive = [3, 19, 20]
            group_indices = [group_indice_face, group_indice_front, group_indice_back,group_indice_exclusive]
            confidence_thresholds = []
            # random down
            # min_ratios = [0.9, 0.75, 0.6, 0.5]
            min_ratios = [0.5, 0.6, 0.65, 0.65]

            # 扫描其中所有confidence 并排序
            for i, indices in enumerate(group_indices):
                group_confidences = []
                for index in indices:
                    for j in range(tea_u_confidence.shape[0]):
                        group_confidences.append(tea_u_confidence[j][index].item())
                group_confidences = sorted(group_confidences, reverse=True)
                cur_ratio = get_current_topkrate(step, args.down_step, min_rate=min_ratios[i])
                sample_nums = len(group_confidences)
                cur_confidence_th = max(group_confidences[min(int(sample_nums * cur_ratio), sample_nums - 1)], 0.2)
                confidence_thresholds.append(cur_confidence_th)

            tea_pseudo_visible = torch.zeros_like(tea_u_confidence)
            for i, indices in enumerate(group_indices):
                for index in indices:
                    tea_pseudo_visible[:, index] = tea_u_confidence[:, index] >= confidence_thresholds[i]
            tea_pseudo_labels = generate_heatmap(coords, tea_pseudo_visible.cpu()).cuda()

            s_loss_l = criterion(s_logits_l, target_heatmaps, target_visible)
            s_loss_pl = criterion(s_logits_u, tea_pseudo_labels, tea_pseudo_visible)
            s_loss = s_loss_l + s_loss_pl

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()

        with amp.autocast(enabled=args.amp):
            t_loss_l = criterion(t_logits_l, target_heatmaps, target_visible)

            # conditional feedback for those sensitive keypoints
            if step >= args.feedback_steps_start:
                with torch.no_grad():
                    s_logits_l_new = student_model(images_l_aug)
                s_loss_l_new = criterion(s_logits_l_new, target_heatmaps, target_visible)
                dot_product = s_loss_l.detach() - s_loss_l_new

                ratio_visible_good = 0.2
                ratio_invisible_bad = 0.2
                tea_u_kps_confidence = torch.clone(tea_u_confidence.detach()).flatten().cpu().numpy()
                _, stu_u_confidence = get_max_preds(s_logits_u)
                stu_u_kps_confidence = torch.clone(stu_u_confidence.detach()).flatten().cpu().numpy()
                tea_u_kps_confidence = sorted(tea_u_kps_confidence, reverse=True)
                stu_u_kps_confidence = sorted(stu_u_kps_confidence, reverse=True)
                min_good_index = int(len(tea_u_kps_confidence) * ratio_visible_good)
                max_bad_index = int(len(stu_u_kps_confidence) * (1 - ratio_invisible_bad))
                min_good_th = max(tea_u_kps_confidence[min_good_index], stu_u_kps_confidence[min_good_index])
                max_bad_th = min(tea_u_kps_confidence[max_bad_index], stu_u_kps_confidence[max_bad_index])
                feedback_vis = torch.clone(tea_u_confidence.detach())
                feedback_vis = torch.where((feedback_vis >= max_bad_th) & (feedback_vis <= min_good_th),
                                           torch.ones_like(feedback_vis), torch.zeros_like(feedback_vis))
                # feedback_vis = torch.where((feedback_vis >= max_bad_th),torch.ones_like(feedback_vis), torch.zeros_like(feedback_vis))
                # feedback hard heatmap
                coords_aug, _ = get_max_preds(t_logits_aug_u)
                t_logits_aug_u_pl = generate_heatmap(coords_aug, feedback_vis.cpu()).cuda()
                feedback_term = criterion(t_logits_aug_u, t_logits_aug_u_pl, feedback_vis)
                t_loss_feedback = dot_product * feedback_term
                feedback_factor = min(1.,(step + 1 - args.feedback_steps_start) /
                                      (args.feedback_steps_complete - args.feedback_steps_start)) * args.feedback_weight
                t_loss = t_loss_l + t_loss_feedback * feedback_factor

            else:
                t_loss = t_loss_l

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        s_optimizer.zero_grad()
        t_optimizer.zero_grad()

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step:4}/{args.total_steps:4}. "
            f"S_LR: {s_optimizer.param_groups[0]['lr']:.5f}. T_LR: {t_optimizer.param_groups[0]['lr']:.5f}. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. T_Loss: {t_losses.avg:.4f}. ")
        pbar.update()

        # evaluate
        if (step + 1) % args.eval_step == 0:
            train_loss = {'Stu_Loss': s_losses.avg, 'Tea_Loss': t_losses.avg}
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalars('Train_Loss', train_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            pbar.close()
            teacher_model.eval()
            student_model.eval()
            ap10k_animalpose_eval_model(cfg, args, step, teacher_model, student_model, t_optimizer, s_optimizer,
                                        t_scheduler,s_scheduler, t_scaler, s_scaler, writer_dict)
            teacher_model.train()
            student_model.train()


def ours_ap10k_animalpose(cfg, args, labeled_loader, unlabeled_loader, teacher_model, student_model,t_optimizer,s_optimizer,
                          t_scheduler, s_scheduler, t_scaler, s_scaler, writer_dict):
    """
    :param writer_dict: writer
    :param args: ..
    :param labeled_loader: ..
    :param unlabeled_loader: ..
    :param teacher_model: ..
    :param student_model: ..
    :param t_optimizer: ..
    :param s_optimizer: ..
    :param t_scheduler: ..
    :param s_scheduler: ..
    :param t_scaler: ..
    :param s_scaler: ..
    :return: none
    """
    # update args info
    program_info_path = os.path.join(args.output_dir, "info", "program_info.txt")
    args.info = "conditional_PL_conditional_feedback"
    args_str = json.dumps(vars(args))
    with open(program_info_path, "w") as f:
        f.write(args_str)
    logger.info(args_str)

    with open(args.keypoints_path, 'r') as f:
        kps_info = json.load(f)
    kps_weights = kps_info['kps_weights']
    criterion = AvgImgMSELoss(kps_weights=kps_weights, num_joints=args.num_joints)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    teacher_model.train()
    student_model.train()

    s_optimizer.zero_grad()
    t_optimizer.zero_grad()

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()

        end = time.time()

        # train
        try:
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except Exception as e:
            # 处理其他特定异常情况
            # 可以输出异常信息等
            logger.error("An error occurred:", e)
            return
        try:
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except Exception as e:
            logger.error("An error occurred:", e)
            return

        data_time.update(time.time() - end)

        images_l_ori = images_l_ori.cuda()
        images_l_aug = images_l_aug.cuda()
        images_u_ori = images_u_ori.cuda()
        images_u_aug = images_u_aug.cuda()

        with amp.autocast(enabled=args.amp):
            label_batch_size = images_l_ori.shape[0]
            images_ori = torch.cat((images_l_ori, images_u_ori)).contiguous()
            images_aug = torch.cat((images_l_aug, images_u_aug)).contiguous()
            # 使teacher model 和 student model 的batch normalization层相似
            with torch.no_grad():
                _ = student_model(images_ori)

            t_logits_aug = teacher_model(images_aug)
            s_logits_aug = student_model(images_aug)
            t_logits_ori = teacher_model(images_ori)

            t_logits_l = t_logits_ori[:label_batch_size]
            t_logits_u = t_logits_ori[label_batch_size:]
            s_logits_l = s_logits_aug[:label_batch_size]
            s_logits_u = s_logits_aug[label_batch_size:]
            t_logits_aug_u = t_logits_aug[label_batch_size:]

            target_heatmaps = torch.stack([t["heatmap"] for t in targets]).cuda(non_blocking=True)
            target_visible = torch.stack([torch.tensor(t["visible"]) for t in targets])
            target_visible[target_visible != 0] = 1

            # semi-supervise 硬伪标签
            coords, tea_u_confidence = get_max_preds(t_logits_u)
            tea_u_confidence = tea_u_confidence.float().squeeze(-1)

            # Conditional PL
            group_indice_face = [0, 1, 2]
            # Ear Re-grouping
            group_indice_front = [5, 6, 7, 8, 9, 10, 17, 18]
            group_indice_back = [4, 11, 12, 13, 14, 15, 16]
            group_indice_exclusive = [3, 19, 20]
            group_indices = [group_indice_face, group_indice_front, group_indice_back,group_indice_exclusive]
            confidence_thresholds = []
            # random down
            # min_ratios = [0.9, 0.75, 0.6, 0.5]
            min_ratios = [0.5, 0.6, 0.65, 0.65]

            # 扫描其中所有confidence 并排序
            for i, indices in enumerate(group_indices):
                group_confidences = []
                for index in indices:
                    for j in range(tea_u_confidence.shape[0]):
                        group_confidences.append(tea_u_confidence[j][index].item())
                group_confidences = sorted(group_confidences, reverse=True)
                cur_ratio = get_current_topkrate(step, args.down_step, min_rate=min_ratios[i])
                sample_nums = len(group_confidences)
                cur_confidence_th = max(group_confidences[min(int(sample_nums * cur_ratio), sample_nums - 1)], 0.2)
                confidence_thresholds.append(cur_confidence_th)

            tea_pseudo_visible = torch.zeros_like(tea_u_confidence)
            for i, indices in enumerate(group_indices):
                for index in indices:
                    tea_pseudo_visible[:, index] = tea_u_confidence[:, index] >= confidence_thresholds[i]
            tea_pseudo_labels = generate_heatmap(coords, tea_pseudo_visible.cpu()).cuda()

            s_loss_l = criterion(s_logits_l, target_heatmaps, target_visible)
            s_loss_pl = criterion(s_logits_u, tea_pseudo_labels, tea_pseudo_visible)
            s_loss = s_loss_l + s_loss_pl

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()

        with amp.autocast(enabled=args.amp):
            t_loss_l = criterion(t_logits_l, target_heatmaps, target_visible)

            # conditional feedback for those sensitive keypoints
            if step >= args.feedback_steps_start:
                with torch.no_grad():
                    s_logits_l_new = student_model(images_l_aug)
                s_loss_l_new = criterion(s_logits_l_new, target_heatmaps, target_visible)
                dot_product = s_loss_l.detach() - s_loss_l_new

                ratio_visible_good = 0.2
                ratio_invisible_bad = 0.2
                tea_u_kps_confidence = torch.clone(tea_u_confidence.detach()).flatten().cpu().numpy()
                _, stu_u_confidence = get_max_preds(s_logits_u)
                stu_u_kps_confidence = torch.clone(stu_u_confidence.detach()).flatten().cpu().numpy()
                tea_u_kps_confidence = sorted(tea_u_kps_confidence, reverse=True)
                stu_u_kps_confidence = sorted(stu_u_kps_confidence, reverse=True)
                min_good_index = int(len(tea_u_kps_confidence) * ratio_visible_good)
                max_bad_index = int(len(stu_u_kps_confidence) * (1 - ratio_invisible_bad))
                min_good_th = max(tea_u_kps_confidence[min_good_index], stu_u_kps_confidence[min_good_index])
                max_bad_th = min(tea_u_kps_confidence[max_bad_index], stu_u_kps_confidence[max_bad_index])
                feedback_vis = torch.clone(tea_u_confidence.detach())
                feedback_vis = torch.where((feedback_vis >= max_bad_th) & (feedback_vis <= min_good_th),
                                           torch.ones_like(feedback_vis), torch.zeros_like(feedback_vis))
                # feedback_vis = torch.where((feedback_vis >= max_bad_th),torch.ones_like(feedback_vis), torch.zeros_like(feedback_vis))
                # feedback hard heatmap
                coords_aug, _ = get_max_preds(t_logits_aug_u)
                t_logits_aug_u_pl = generate_heatmap(coords_aug, feedback_vis.cpu()).cuda()
                feedback_term = criterion(t_logits_aug_u, t_logits_aug_u_pl, feedback_vis)
                t_loss_feedback = dot_product * feedback_term
                feedback_factor = min(1.,(step + 1 - args.feedback_steps_start) /
                                      (args.feedback_steps_complete - args.feedback_steps_start)) * args.feedback_weight
                t_loss = t_loss_l + t_loss_feedback * feedback_factor

            else:
                t_loss = t_loss_l

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        s_optimizer.zero_grad()
        t_optimizer.zero_grad()

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step:4}/{args.total_steps:4}. "
            f"S_LR: {s_optimizer.param_groups[0]['lr']:.5f}. T_LR: {t_optimizer.param_groups[0]['lr']:.5f}. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. T_Loss: {t_losses.avg:.4f}. ")
        pbar.update()

        # evaluate
        if (step + 1) % args.eval_step == 0:
            train_loss = {'Stu_Loss': s_losses.avg, 'Tea_Loss': t_losses.avg}
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalars('Train_Loss', train_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            pbar.close()
            teacher_model.eval()
            student_model.eval()
            ap10k_animalpose_eval_model(cfg, args, step, teacher_model, student_model, t_optimizer, s_optimizer,
                                        t_scheduler,s_scheduler, t_scaler, s_scaler, writer_dict)
            teacher_model.train()
            student_model.train()


def finetune(cfg, args, labeled_loader, unlabeled_loader, teacher_model, student_model,t_optimizer,s_optimizer,
             t_scheduler, s_scheduler, t_scaler, s_scaler, writer_dict):
    """
    :param writer_dict: writer
    :param args: ..
    :param labeled_loader: ..
    :param unlabeled_loader: ..
    :param teacher_model: ..
    :param student_model: ..
    :param t_optimizer: ..
    :param s_optimizer: ..
    :param t_scheduler: ..
    :param s_scheduler: ..
    :param t_scaler: ..
    :param s_scaler: ..
    :return: none
    """
    # update args info
    program_info_path = os.path.join(args.output_dir, "info", "program_info.txt")
    args.info = "finetune"
    args_str = json.dumps(vars(args))
    with open(program_info_path, "w") as f:
        f.write(args_str)
    logger.info(args_str)

    with open(args.keypoints_path, 'r') as f:
        kps_info = json.load(f)
    kps_weights = kps_info['kps_weights']
    criterion = AvgImgMSELoss(kps_weights=kps_weights, num_joints=args.num_joints)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    teacher_model.train()
    student_model.train()

    s_optimizer.zero_grad()
    t_optimizer.zero_grad()

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()

        end = time.time()

        # train
        try:
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except Exception as e:
            # 处理其他特定异常情况
            # 可以输出异常信息等
            logger.error("An error occurred:", e)
            return
        try:
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except Exception as e:
            logger.error("An error occurred:", e)
            return

        data_time.update(time.time() - end)

        images_l_ori = images_l_ori.cuda()
        images_l_aug = images_l_aug.cuda()
        images_u_ori = images_u_ori.cuda()
        images_u_aug = images_u_aug.cuda()

        with amp.autocast(enabled=args.amp):
            label_batch_size = images_l_ori.shape[0]
            images_ori = torch.cat((images_l_ori, images_u_ori)).contiguous()
            images_aug = torch.cat((images_l_aug, images_u_aug)).contiguous()
            # 使teacher model 和 student model 的batch normalization层相似
            with torch.no_grad():
                _ = student_model(images_ori)

            t_logits_aug = teacher_model(images_aug)
            s_logits_aug = student_model(images_aug)
            t_logits_ori = teacher_model(images_ori)

            t_logits_l = t_logits_ori[:label_batch_size]
            t_logits_u = t_logits_ori[label_batch_size:]
            s_logits_l = s_logits_aug[:label_batch_size]
            s_logits_u = s_logits_aug[label_batch_size:]
            t_logits_aug_u = t_logits_aug[label_batch_size:]

            target_heatmaps = torch.stack([t["heatmap"] for t in targets]).cuda(non_blocking=True)
            target_visible = torch.stack([torch.tensor(t["visible"]) for t in targets])
            target_visible[target_visible != 0] = 1

            # semi-supervise 硬伪标签
            coords, tea_u_confidence = get_max_preds(t_logits_u)
            tea_u_confidence = tea_u_confidence.float().squeeze(-1)

            # Conditional PL
            group_indice_face = [0, 1, 2]
            group_indice_front = [5, 6, 7, 8, 9, 10]
            group_indice_back = [4, 11, 12, 13, 14, 15, 16]
            group_indice_exclusive = [3]
            group_indices = [group_indice_face, group_indice_front, group_indice_back,group_indice_exclusive]
            confidence_thresholds = []
            # random down
            # min_ratios = [0.9, 0.75, 0.6, 0.5]
            min_ratios = [0.5, 0.6, 0.65, 0.65]
            # 扫描其中所有confidence 并排序，记录第60%大的confidence
            for i, indices in enumerate(group_indices):
                group_confidences = []
                for index in indices:
                    for j in range(tea_u_confidence.shape[0]):
                        group_confidences.append(tea_u_confidence[j][index].item())
                group_confidences = sorted(group_confidences, reverse=True)
                cur_ratio = get_current_topkrate(step, args.down_step, min_rate=min_ratios[i])
                sample_nums = len(group_confidences)
                cur_confidence_th = max(group_confidences[min(int(sample_nums * cur_ratio), sample_nums - 1)], 0.2)
                confidence_thresholds.append(cur_confidence_th)

            tea_pseudo_visible = torch.zeros_like(tea_u_confidence)
            for i, indices in enumerate(group_indices):
                for index in indices:
                    tea_pseudo_visible[:, index] = tea_u_confidence[:, index] >= confidence_thresholds[i]
            tea_pseudo_labels = generate_heatmap(coords, tea_pseudo_visible.cpu()).cuda()

            s_loss_l = criterion(s_logits_l, target_heatmaps, target_visible)
            s_loss_pl = criterion(s_logits_u, tea_pseudo_labels, tea_pseudo_visible)
            s_loss = s_loss_l + s_loss_pl

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()

        with amp.autocast(enabled=args.amp):
            t_loss_l = criterion(t_logits_l, target_heatmaps, target_visible)

            # conditional feedback for those sensitive keypoints
            if step >= args.feedback_steps_start:
                with torch.no_grad():
                    s_logits_l_new = student_model(images_l_aug)
                s_loss_l_new = criterion(s_logits_l_new, target_heatmaps, target_visible)
                dot_product = s_loss_l.detach() - s_loss_l_new

                ratio_visible_good = 0.2
                ratio_invisible_bad = 0.2
                tea_u_kps_confidence = torch.clone(tea_u_confidence.detach()).flatten().cpu().numpy()
                _, stu_u_confidence = get_max_preds(s_logits_u)
                stu_u_kps_confidence = torch.clone(stu_u_confidence.detach()).flatten().cpu().numpy()
                tea_u_kps_confidence = sorted(tea_u_kps_confidence, reverse=True)
                stu_u_kps_confidence = sorted(stu_u_kps_confidence, reverse=True)
                min_good_index = int(len(tea_u_kps_confidence) * ratio_visible_good)
                max_bad_index = int(len(stu_u_kps_confidence) * (1 - ratio_invisible_bad))
                min_good_th = max(tea_u_kps_confidence[min_good_index], stu_u_kps_confidence[min_good_index])
                max_bad_th = min(tea_u_kps_confidence[max_bad_index], stu_u_kps_confidence[max_bad_index])
                feedback_vis = torch.clone(tea_u_confidence.detach())
                feedback_vis = torch.where((feedback_vis >= max_bad_th) & (feedback_vis <= min_good_th),
                                           torch.ones_like(feedback_vis), torch.zeros_like(feedback_vis))
                # feedback_vis = torch.where((feedback_vis >= max_bad_th),torch.ones_like(feedback_vis), torch.zeros_like(feedback_vis))
                # feedback hard heatmap
                coords_aug, _ = get_max_preds(t_logits_aug_u)
                t_logits_aug_u_pl = generate_heatmap(coords_aug, feedback_vis.cpu()).cuda()
                feedback_term = criterion(t_logits_aug_u, t_logits_aug_u_pl, feedback_vis)
                t_loss_feedback = dot_product * feedback_term
                feedback_factor = min(1.,(step + 1 - args.feedback_steps_start) /
                                      (args.feedback_steps_complete - args.feedback_steps_start)) * args.feedback_weight
                t_loss = t_loss_l + t_loss_feedback * feedback_factor

            else:
                t_loss = t_loss_l

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        s_optimizer.zero_grad()
        t_optimizer.zero_grad()

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step:4}/{args.total_steps:4}. "
            f"S_LR: {s_optimizer.param_groups[0]['lr']:.5f}. T_LR: {t_optimizer.param_groups[0]['lr']:.5f}. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. T_Loss: {t_losses.avg:.4f}. ")
        pbar.update()

        # evaluate
        if (step + 1) % args.eval_step == 0:
            train_loss = {'Stu_Loss': s_losses.avg, 'Tea_Loss': t_losses.avg}
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalars('Train_Loss', train_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            pbar.close()
            teacher_model.eval()
            student_model.eval()
            ap10k_eval_model(cfg, args, step, teacher_model, student_model, t_optimizer, s_optimizer,
                                      t_scheduler,s_scheduler, t_scaler, s_scaler, writer_dict)
            teacher_model.train()
            student_model.train()


def sl_finetune(cfg, args, labeled_loader, unlabeled_loader, teacher_model, student_model,t_optimizer,s_optimizer,
                t_scheduler, s_scheduler, t_scaler, s_scaler, writer_dict):
    """
    :param writer_dict: writer
    :param args: ..
    :param labeled_loader: ..
    :param unlabeled_loader: ..
    :param teacher_model: ..
    :param student_model: ..
    :param t_optimizer: ..
    :param s_optimizer: ..
    :param t_scheduler: ..
    :param s_scheduler: ..
    :param t_scaler: ..
    :param s_scaler: ..
    :return: none
    """
    # update args info
    program_info_path = os.path.join(args.output_dir, "info", "program_info.txt")
    args.info = "sl_finetune"
    args_str = json.dumps(vars(args))
    with open(program_info_path, "w") as f:
        f.write(args_str)
    logger.info(args_str)

    with open(args.keypoints_path, 'r') as f:
        kps_info = json.load(f)
    kps_weights = kps_info['kps_weights']
    criterion = AvgImgMSELoss(kps_weights=kps_weights, num_joints=args.num_joints)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    teacher_model.train()
    student_model.train()

    s_optimizer.zero_grad()
    t_optimizer.zero_grad()

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()

        end = time.time()

        # train
        try:
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except Exception as e:
            # 处理其他特定异常情况
            # 可以输出异常信息等
            logger.error("An error occurred:", e)
            return
        try:
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except Exception as e:
            logger.error("An error occurred:", e)
            return

        data_time.update(time.time() - end)

        images_l_ori = images_l_ori.cuda()
        images_l_aug = images_l_aug.cuda()
        images_u_ori = images_u_ori.cuda()
        images_u_aug = images_u_aug.cuda()

        with amp.autocast(enabled=args.amp):
            label_batch_size = images_l_ori.shape[0]
            images_ori = torch.cat((images_l_ori, images_u_ori)).contiguous()
            images_aug = torch.cat((images_l_aug, images_u_aug)).contiguous()
            # 使teacher model 和 student model 的batch normalization层相似
            with torch.no_grad():
                _ = student_model(images_ori)

            t_logits_aug = teacher_model(images_aug)
            s_logits_aug = student_model(images_aug)
            t_logits_ori = teacher_model(images_ori)

            t_logits_l = t_logits_ori[:label_batch_size]
            t_logits_u = t_logits_ori[label_batch_size:]
            s_logits_l = s_logits_aug[:label_batch_size]
            s_logits_u = s_logits_aug[label_batch_size:]
            t_logits_aug_u = t_logits_aug[label_batch_size:]

            target_heatmaps = torch.stack([t["heatmap"] for t in targets]).cuda(non_blocking=True)
            target_visible = torch.stack([torch.tensor(t["visible"]) for t in targets])
            target_visible[target_visible != 0] = 1

            # semi-supervise 硬伪标签
            coords, tea_u_confidence = get_max_preds(t_logits_u)
            tea_u_confidence = tea_u_confidence.float().squeeze(-1)

            # Conditional PL
            group_indice_face = [0, 1, 2]
            group_indice_front = [5, 6, 7, 8, 9, 10]
            group_indice_back = [4, 11, 12, 13, 14, 15, 16]
            group_indice_exclusive = [3]
            group_indices = [group_indice_face, group_indice_front, group_indice_back,group_indice_exclusive]
            confidence_thresholds = []
            # random down
            # min_ratios = [0.9, 0.75, 0.6, 0.5]
            min_ratios = [0.5, 0.6, 0.65, 0.65]
            # 扫描其中所有confidence 并排序，记录第60%大的confidence
            for i, indices in enumerate(group_indices):
                group_confidences = []
                for index in indices:
                    for j in range(tea_u_confidence.shape[0]):
                        group_confidences.append(tea_u_confidence[j][index].item())
                group_confidences = sorted(group_confidences, reverse=True)
                cur_ratio = get_current_topkrate(step, args.down_step, min_rate=min_ratios[i])
                sample_nums = len(group_confidences)
                cur_confidence_th = max(group_confidences[min(int(sample_nums * cur_ratio), sample_nums - 1)], 0.2)
                confidence_thresholds.append(cur_confidence_th)

            tea_pseudo_visible = torch.zeros_like(tea_u_confidence)
            for i, indices in enumerate(group_indices):
                for index in indices:
                    tea_pseudo_visible[:, index] = tea_u_confidence[:, index] >= confidence_thresholds[i]
            tea_pseudo_labels = generate_heatmap(coords, tea_pseudo_visible.cpu()).cuda()

            s_loss_l = criterion(s_logits_l, target_heatmaps, target_visible)
            s_loss_pl = criterion(s_logits_u, tea_pseudo_labels, tea_pseudo_visible)
            s_loss = s_loss_l + s_loss_pl

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()

        with amp.autocast(enabled=args.amp):
            t_loss_l = criterion(t_logits_l, target_heatmaps, target_visible)
            t_loss = t_loss_l

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        s_optimizer.zero_grad()
        t_optimizer.zero_grad()

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step:4}/{args.total_steps:4}. "
            f"S_LR: {s_optimizer.param_groups[0]['lr']:.5f}. T_LR: {t_optimizer.param_groups[0]['lr']:.5f}. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. T_Loss: {t_losses.avg:.4f}. ")
        pbar.update()

        # evaluate
        if (step + 1) % args.eval_step == 0:
            train_loss = {'Stu_Loss': s_losses.avg, 'Tea_Loss': t_losses.avg}
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalars('Train_Loss', train_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            pbar.close()
            teacher_model.eval()
            student_model.eval()
            ap10k_eval_model(cfg, args, step, teacher_model, student_model, t_optimizer, s_optimizer,
                                      t_scheduler,s_scheduler, t_scaler, s_scaler, writer_dict)
            teacher_model.train()
            student_model.train()


def uda(cfg, args, labeled_loader, unlabeled_loader, teacher_model, student_model,t_optimizer,s_optimizer,
        t_scheduler, s_scheduler, t_scaler, s_scaler, writer_dict):
    """
    :param writer_dict: writer
    :param args: ..
    :param labeled_loader: ..
    :param unlabeled_loader: ..
    :param teacher_model: ..
    :param student_model: ..
    :param t_optimizer: ..
    :param s_optimizer: ..
    :param t_scheduler: ..
    :param s_scheduler: ..
    :param t_scaler: ..
    :param s_scaler: ..
    :return: none
    """
    # update args info
    program_info_path = os.path.join(args.output_dir, "info", "program_info.txt")
    args.info = "uda"
    args_str = json.dumps(vars(args))
    with open(program_info_path, "w") as f:
        f.write(args_str)
    logger.info(args_str)

    with open(args.keypoints_path, 'r') as f:
        kps_info = json.load(f)
    kps_weights = kps_info['kps_weights']
    criterion = AvgImgMSELoss(kps_weights=kps_weights, num_joints=args.num_joints)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    teacher_model.train()
    student_model.train()

    s_optimizer.zero_grad()
    t_optimizer.zero_grad()

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()

        end = time.time()

        # train
        try:
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except Exception as e:
            # 处理其他特定异常情况
            # 可以输出异常信息等
            logger.error("An error occurred:", e)
            return
        try:
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except Exception as e:
            logger.error("An error occurred:", e)
            return

        data_time.update(time.time() - end)

        images_l_ori = images_l_ori.cuda()
        images_l_aug = images_l_aug.cuda()
        images_u_ori = images_u_ori.cuda()
        images_u_aug = images_u_aug.cuda()

        with amp.autocast(enabled=args.amp):
            label_batch_size = images_l_ori.shape[0]
            images_ori = torch.cat((images_l_ori, images_u_ori)).contiguous()
            images_aug = torch.cat((images_l_aug, images_u_aug)).contiguous()
            # 使teacher model 和 student model 的batch normalization层相似
            with torch.no_grad():
                _ = student_model(images_ori)

            s_logits_aug = student_model(images_aug)
            t_logits_ori = teacher_model(images_ori)

            flipped_images = transforms.flip_images(images_aug)
            flipped_outputs = teacher_model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, kps_info["flip_pairs"])
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]

            t_logits_l = t_logits_ori[:label_batch_size]
            t_logits_u = t_logits_ori[label_batch_size:]
            s_logits_l = s_logits_aug[:label_batch_size]
            s_logits_u = s_logits_aug[label_batch_size:]
            t_logits_aug_u = flipped_outputs[label_batch_size:]

            target_heatmaps = torch.stack([t["heatmap"] for t in targets]).cuda(non_blocking=True)
            target_visible = torch.stack([torch.tensor(t["visible"]) for t in targets])
            target_visible[target_visible != 0] = 1

            # semi-supervise 硬伪标签
            coords, tea_u_confidence = get_max_preds(t_logits_u)
            tea_u_confidence = (tea_u_confidence > 0.4).float().squeeze(-1)

            tea_pseudo_labels = generate_heatmap(coords, tea_u_confidence.cpu()).cuda()

            s_loss_l = criterion(s_logits_l, target_heatmaps, target_visible)
            s_loss_pl = criterion(s_logits_u, tea_pseudo_labels, tea_u_confidence)
            s_loss = s_loss_l + s_loss_pl

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()

        with amp.autocast(enabled=args.amp):
            t_loss_l = criterion(t_logits_l, target_heatmaps, target_visible)
            uda_weights = min(1.,step / args.uda_steps)
            t_loss_uda = criterion(t_logits_aug_u,t_logits_u.detach(),tea_u_confidence)

            t_loss = t_loss_l + t_loss_uda * uda_weights

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        s_optimizer.zero_grad()
        t_optimizer.zero_grad()

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step:4}/{args.total_steps:4}. "
            f"S_LR: {s_optimizer.param_groups[0]['lr']:.5f}. T_LR: {t_optimizer.param_groups[0]['lr']:.5f}. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. T_Loss: {t_losses.avg:.4f}. ")
        pbar.update()

        # evaluate
        if (step + 1) % args.eval_step == 0:
            train_loss = {'Stu_Loss': s_losses.avg, 'Tea_Loss': t_losses.avg}
            if writer_dict is not None:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalars('Train_Loss', train_loss, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

            pbar.close()
            teacher_model.eval()
            student_model.eval()
            ap10k_eval_model(cfg, args, step, teacher_model, student_model, t_optimizer, s_optimizer,
                                      t_scheduler,s_scheduler, t_scaler, s_scaler, writer_dict)
            teacher_model.train()
            student_model.train()


def mpl(cfg, args, labeled_loader, unlabeled_loader, teacher_model, student_model,t_optimizer,s_optimizer,
        t_scheduler, s_scheduler, t_scaler, s_scaler, writer_dict):
    """
    :param writer_dict: writer
    :param args: ..
    :param labeled_loader: ..
    :param unlabeled_loader: ..
    :param teacher_model: ..
    :param student_model: ..
    :param t_optimizer: ..
    :param s_optimizer: ..
    :param t_scheduler: ..
    :param s_scheduler: ..
    :param t_scaler: ..
    :param s_scaler: ..
    :return: none
    """
    # update args info
    program_info_path = os.path.join(args.output_dir, "info", "program_info.txt")
    args.info = "mpl"
    args_str = json.dumps(vars(args))
    with open(program_info_path, "w") as f:
        f.write(args_str)
    logger.info(args_str)

    with open(args.keypoints_path, 'r') as f:
        kps_info = json.load(f)
    kps_weights = kps_info['kps_weights']
    criterion = AvgImgMSELoss(kps_weights=kps_weights, num_joints=args.num_joints)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    teacher_model.train()
    student_model.train()

    s_optimizer.zero_grad()
    t_optimizer.zero_grad()

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()

        end = time.time()

        # train
        try:
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except Exception as e:
            # 处理其他特定异常情况
            # 可以输出异常信息等
            logger.error("An error occurred:", e)
            return
        try:
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except Exception as e:
            logger.error("An error occurred:", e)
            return

        data_time.update(time.time() - end)

        images_l_ori = images_l_ori.cuda()
        images_l_aug = images_l_aug.cuda()
        images_u_ori = images_u_ori.cuda()
        images_u_aug = images_u_aug.cuda()

        with amp.autocast(enabled=args.amp):
            label_batch_size = images_l_ori.shape[0]
            images_ori = torch.cat((images_l_ori, images_u_ori)).contiguous()
            images_aug = torch.cat((images_l_aug, images_u_aug)).contiguous()
            # 使teacher model 和 student model 的batch normalization层相似
            with torch.no_grad():
                _ = student_model(images_ori)

            t_logits_aug = teacher_model(images_aug)
            s_logits_aug = student_model(images_aug)
            t_logits_ori = teacher_model(images_ori)

            t_logits_l = t_logits_ori[:label_batch_size]
            t_logits_u = t_logits_ori[label_batch_size:]
            s_logits_l = s_logits_aug[:label_batch_size]
            s_logits_u = s_logits_aug[label_batch_size:]
            t_logits_aug_u = t_logits_aug[label_batch_size:]

            target_heatmaps = torch.stack([t["heatmap"] for t in targets]).cuda(non_blocking=True)
            target_visible = torch.stack([torch.tensor(t["visible"]) for t in targets])
            target_visible[target_visible != 0] = 1

            # semi-supervise 硬伪标签
            coords, tea_u_confidence = get_max_preds(t_logits_u)
            tea_u_confidence = (tea_u_confidence > 0.2).float().squeeze(-1)
            tea_pseudo_labels = generate_heatmap(coords, tea_u_confidence.cpu()).cuda()

            s_loss_l = criterion(s_logits_l, target_heatmaps, target_visible)
            s_loss_pl = criterion(s_logits_u, tea_pseudo_labels, tea_u_confidence)
            s_loss = s_loss_l + s_loss_pl

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()

        with amp.autocast(enabled=args.amp):
            t_loss_l = criterion(t_logits_l, target_heatmaps, target_visible)
            # conditional feedback for those sensitive keypoints
            with torch.no_grad():
                s_logits_l_new = student_model(images_l_aug)
            s_loss_l_new = criterion(s_logits_l_new, target_heatmaps, target_visible)
            dot_product = s_loss_l.detach() - s_loss_l_new

            # feedback hard heatmap
            coords_aug, feedback_vis = get_max_preds(t_logits_aug_u)
            feedback_vis = (feedback_vis > 0.2).float().squeeze(-1)
            t_logits_aug_u_pl = generate_heatmap(coords_aug, feedback_vis.cpu()).cuda()
            feedback_term = criterion(t_logits_aug_u, t_logits_aug_u_pl, feedback_vis)
            t_loss_feedback = dot_product * feedback_term
            t_loss = t_loss_l + t_loss_feedback

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        s_optimizer.zero_grad()
        t_optimizer.zero_grad()

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step:4}/{args.total_steps:4}. "
            f"S_LR: {s_optimizer.param_groups[0]['lr']:.5f}. T_LR: {t_optimizer.param_groups[0]['lr']:.5f}. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. T_Loss: {t_losses.avg:.4f}. ")
        pbar.update()

        # evaluate
        if (step + 1) % args.eval_step == 0:
            train_loss = {'Stu_Loss': s_losses.avg, 'Tea_Loss': t_losses.avg}
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalars('Train_Loss', train_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            pbar.close()
            teacher_model.eval()
            student_model.eval()
            ap10k_eval_model(cfg, args, step, teacher_model, student_model, t_optimizer, s_optimizer,
                                      t_scheduler,s_scheduler, t_scaler, s_scaler, writer_dict)
            teacher_model.train()
            student_model.train()


def fixmatch(cfg, args, labeled_loader, unlabeled_loader, model, optimizer, scheduler, scaler, writer_dict):
    """
    :param cfg:
    :param args:
    :param labeled_loader:
    :param unlabeled_loader:
    :param model:
    :param optimizer:
    :param scheduler:
    :param scaler:
    :param writer_dict:
    :return:
    """
    # update args info
    program_info_path = os.path.join(args.output_dir, "info", "program_info.txt")
    args.info = "setting_fixmatch"
    args_str = json.dumps(vars(args))
    with open(program_info_path, "w") as f:
        f.write(args_str)
    logger.info(args_str)

    with open(args.keypoints_path, 'r') as f:
        kps_info = json.load(f)
    kps_weights = kps_info['kps_weights']
    criterion = AvgImgMSELoss(kps_weights=kps_weights, num_joints=args.num_joints)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    model.train()
    optimizer.zero_grad()

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()

        end = time.time()

        # train
        try:
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            (images_l_ori, images_l_aug), targets = next(labeled_iter)
        except Exception as e:
            # 处理其他特定异常情况
            # 可以输出异常信息等
            logger.error("An error occurred:", e)
            return
        try:
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            (images_u_ori, images_u_aug), _ = next(unlabeled_iter)
        except Exception as e:
            logger.error("An error occurred:", e)
            return

        data_time.update(time.time() - end)

        images_l_ori = images_l_ori.cuda()
        images_l_aug = images_l_aug.cuda()
        images_u_ori = images_u_ori.cuda()
        images_u_aug = images_u_aug.cuda()

        with amp.autocast(enabled=args.amp):
            label_batch_size = images_l_ori.shape[0]
            images_ori = torch.cat((images_l_ori, images_u_ori)).contiguous()
            images_aug = torch.cat((images_l_aug, images_u_aug)).contiguous()

            logits_ori = model(images_ori)
            flipped_images = transforms.flip_images(images_aug[label_batch_size:])
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, kps_info["flip_pairs"])
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]

            logits_ori_l = logits_ori[:label_batch_size]
            logits_ori_u = logits_ori[label_batch_size:]

            target_heatmaps = torch.stack([t["heatmap"] for t in targets]).cuda(non_blocking=True)
            target_visible = torch.stack([torch.tensor(t["visible"]) for t in targets])
            target_visible[target_visible != 0] = 1

            # semi-supervise 硬伪标签
            coords, u_confidence = get_max_preds(logits_ori_u)
            u_confidence = (u_confidence > 0.4).float().squeeze(-1)
            pseudo_labels = generate_heatmap(coords, u_confidence.cpu()).cuda()

            loss_l = criterion(logits_ori_l, target_heatmaps, target_visible)
            loss_u = criterion(flipped_outputs, pseudo_labels, u_confidence)
            loss = loss_l + loss_u

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        losses.update(loss.item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step:4}/{args.total_steps:4}. "
            f"LR: {optimizer.param_groups[0]['lr']:.5f}. "
            f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}.")
        pbar.update()

        # evaluate
        if (step + 1) % args.eval_step == 0:
            train_loss = {'Train_Loss': losses.avg}
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalars('Train_Loss', train_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            pbar.close()
            model.eval()
            ap10k_eval_model_single(cfg,args,step,model,optimizer,scheduler,scaler,writer_dict)
            model.train()


def eval_from_scarcenet(cfg,model,output_dir):
    # for mix dataset
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=True).cuda()

    valid_dataset = dataset_animal.ap10k_animalpose(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.VAL_SET, False,
        tf.Compose([
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    oks_val, pck_val = validate(cfg, valid_loader, valid_dataset, model, criterion, output_dir, animalpose=True)
    return oks_val, pck_val


def eval_ap10k(cfg,model,output_dir):
    # for mix dataset
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=True).cuda()

    valid_dataset = dataset_animal.ap10k(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.VAL_SET, False,
        tf.Compose([
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    oks_val, pck_val = validate(cfg, valid_loader, valid_dataset, model, criterion, output_dir, animalpose=True)
    return oks_val, pck_val


def ap10k_animalpose_eval_model(cfg, args, step, teacher_model, student_model, t_optimizer, s_optimizer, t_scheduler,
                                    s_scheduler, t_scaler, s_scaler, writer_dict):
    """
        For multi GPUs
        This function directly use teacher model and student model themselves to validate
        :param writer_dict: Summary Writer of TensorboardX
        :param args: ...
        :param step: current training step -> epoch
        :param teacher_model: teacher model on multiGPUs
        :param student_model: student model on multiGPUs
        :param t_optimizer: used to save info
        :param s_optimizer: used to save info
        :param t_scheduler: used to save info
        :param s_scheduler: used to save info
        :param t_scaler: used to save info
        :param s_scaler: used to save info
        :param val_dataset: validating dataset.Here is AP-10K Val + Animal Pose Val
        :return: None
    """
    epoch = step // args.eval_step
    save_files = {
        'teacher_model': teacher_model.module.state_dict() if hasattr(teacher_model,
                                                                      'module') else teacher_model.state_dict(),
        'student_model': student_model.module.state_dict() if hasattr(student_model,
                                                                      'module') else student_model.state_dict(),
        'teacher_optimizer': t_optimizer.state_dict(),
        'student_optimizer': s_optimizer.state_dict(),
        'teacher_scheduler': t_scheduler.state_dict(),
        'student_scheduler': s_scheduler.state_dict(),
        'step': step,
        'epoch': epoch}
    if args.amp:
        save_files["teacher_scaler"] = t_scaler.state_dict()
        save_files["student_scaler"] = s_scaler.state_dict()

    torch.save(save_files, "{}/checkpoint.pth".format(args.output_dir))

    stu_oks, stu_pck = eval_from_scarcenet(cfg,student_model,args.output_dir)
    tea_oks, tea_pck = eval_from_scarcenet(cfg,teacher_model,args.output_dir)

    if stu_oks > args.best_oks:
        args.best_oks = stu_oks
        args.best_oks_epoch = epoch
        torch.save(save_files['student_model'], "{}/best-oks.pth".format(args.output_dir))
    if stu_pck > args.best_pck:
        args.best_pck = stu_pck
        args.best_pck_epoch = epoch
        torch.save(save_files['student_model'], "{}/best-pck.pth".format(args.output_dir))

    oks_dict = {'Stu_OKS': stu_oks, 'Tea_OKS': tea_oks}
    pck_dict = {'Stu_PCK': stu_pck, 'Tea_PCK': tea_pck}

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalars('Val_OKS', oks_dict, global_steps)
    writer.add_scalars('Val_PCK', pck_dict, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1

    # write into txt
    val_path = os.path.join(args.output_dir, "info/val_log.txt")
    with open(val_path, "a") as f:
        # 写入的数据包括coco指标还有loss和learning rate
        result_info = [
            f"student_oks:{stu_oks:.6f}",
            f"student_pck:{stu_pck:.6f}",
            f"teacher_oks:{tea_oks:.6f}",
            f"teacher_pck:{tea_pck:.6f}"
        ]
        txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
        f.write(txt + "\n")
        logger.info(txt)


def ap10k_eval_model_single(cfg, args, step, model, optimizer, scheduler, scaler, writer_dict):
    """
    :param cfg:
    :param args:
    :param step:
    :param model:
    :param optimizer:
    :param scheduler:
    :param scaler:
    :param writer_dict:
    :return:
    """
    epoch = step // args.eval_step
    save_files = {
        'model': model.module.state_dict() if hasattr(model,'module') else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
        'epoch': epoch}
    if args.amp:
        save_files["scaler"] = scaler.state_dict()

    torch.save(save_files, "{}/checkpoint.pth".format(args.output_dir))

    oks, pck = eval_ap10k(cfg,model,args.output_dir)

    if oks > args.best_oks:
        args.best_oks = oks
        args.best_oks_epoch = epoch
        torch.save(save_files['model'], "{}/best-oks.pth".format(args.output_dir))
    if pck > args.best_pck:
        args.best_pck = pck
        args.best_pck_epoch = epoch
        torch.save(save_files['model'], "{}/best-pck.pth".format(args.output_dir))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('Val_OKS', oks, global_steps)
    writer.add_scalar('Val_PCK', pck, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1

    # write into txt
    val_path = os.path.join(args.output_dir, "info/val_log.txt")
    with open(val_path, "a") as f:
        # 写入的数据包括coco指标还有loss和learning rate
        result_info = [
            f"oks:{oks:.6f}",
            f"pck:{pck:.6f}"
        ]
        txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
        f.write(txt + "\n")
        logger.info(txt)


def ap10k_eval_model(cfg, args, step, teacher_model, student_model, t_optimizer, s_optimizer, t_scheduler,s_scheduler,
                     t_scaler, s_scaler, writer_dict):
    """
        For multi GPUs
        This function directly use teacher model and student model themselves to validate
        :param writer_dict: Summary Writer of TensorboardX
        :param args: ...
        :param step: current training step -> epoch
        :param teacher_model: teacher model on multiGPUs
        :param student_model: student model on multiGPUs
        :param t_optimizer: used to save info
        :param s_optimizer: used to save info
        :param t_scheduler: used to save info
        :param s_scheduler: used to save info
        :param t_scaler: used to save info
        :param s_scaler: used to save info
        :return: None
    """
    epoch = step // args.eval_step
    save_files = {
        'teacher_model': teacher_model.module.state_dict() if hasattr(teacher_model,
                                                                      'module') else teacher_model.state_dict(),
        'student_model': student_model.module.state_dict() if hasattr(student_model,
                                                                      'module') else student_model.state_dict(),
        'teacher_optimizer': t_optimizer.state_dict(),
        'student_optimizer': s_optimizer.state_dict(),
        'teacher_scheduler': t_scheduler.state_dict(),
        'student_scheduler': s_scheduler.state_dict(),
        'step': step,
        'epoch': epoch}
    if args.amp:
        save_files["teacher_scaler"] = t_scaler.state_dict()
        save_files["student_scaler"] = s_scaler.state_dict()

    torch.save(save_files, "{}/checkpoint.pth".format(args.output_dir))

    stu_oks, stu_pck = eval_ap10k(cfg,student_model,args.output_dir)
    tea_oks, tea_pck = eval_ap10k(cfg,teacher_model,args.output_dir)

    if stu_oks > args.best_oks:
        args.best_oks = stu_oks
        args.best_oks_epoch = epoch
        torch.save(save_files['student_model'], "{}/best-oks.pth".format(args.output_dir))
    if stu_pck > args.best_pck:
        args.best_pck = stu_pck
        args.best_pck_epoch = epoch
        torch.save(save_files['student_model'], "{}/best-pck.pth".format(args.output_dir))

    oks_dict = {'Stu_OKS': stu_oks, 'Tea_OKS': tea_oks}
    pck_dict = {'Stu_PCK': stu_pck, 'Tea_PCK': tea_pck}

    if writer_dict is not None:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalars('Val_OKS', oks_dict, global_steps)
        writer.add_scalars('Val_PCK', pck_dict, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    # write into txt
    val_path = os.path.join(args.output_dir, "info/val_log.txt")
    with open(val_path, "a") as f:
        # 写入的数据包括coco指标还有loss和learning rate
        result_info = [
            f"student_oks:{stu_oks:.6f}",
            f"student_pck:{stu_pck:.6f}",
            f"teacher_oks:{tea_oks:.6f}",
            f"teacher_pck:{tea_pck:.6f}"
        ]
        txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
        f.write(txt + "\n")
        logger.info(txt)
