import os.path
import random
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader, RandomSampler

from train_utils import transforms
from train_utils.dataset import CocoKeypoint, MixKeypoint
from models.hrnet import HighResolutionNet
from train_utils.transforms import get_max_preds


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gt_show(dataset,num_joints):
    keypoints_path = f"../info/{dataset}_keypoints_format.json"
    with open(keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    fixed_size = (256,256)
    heatmap_hw = (fixed_size[0] // 4, fixed_size[1] // 4)
    kps_weights = np.array(animal_kps_info["kps_weights"],
                           dtype=np.float32).reshape((num_joints,))

    data_transform = {
        "val": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    label_data_root = f"../../dataset/{dataset}"
    train_dataset = CocoKeypoint(root=label_data_root, dataset=dataset,mode="train",transform=data_transform["val"],
                                 fixed_size=fixed_size, data_type="keypoints")
    train_dataset.animal_gt_show(animal_category=None,shuffle=True)


def dt_show():
    keypoints_path = f"../info/keypoints_definition.json"
    with open(keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    fixed_size = (256,256)
    heatmap_hw = (fixed_size[0] // 4, fixed_size[1] // 4)
    kps_weights = np.array(animal_kps_info["kps_weights"],
                           dtype=np.float32).reshape((26,))
    data_transform = {
        "val": transforms.Compose([
            transforms.LabelFormatTrans(extend_flag=True),
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    model_a = HighResolutionNet(num_joints=26)
    model_b = HighResolutionNet(num_joints=26)
    model_c = HighResolutionNet(num_joints=26)
    model_d = HighResolutionNet(num_joints=26)
    path_a = '../save_weights/mix_0.1_SL.pth'
    path_b = '../save_weights/mix_0.1_ssl.pth'
    path_c = '../save_weights/mix_0.1_self.pth'
    path_d = '../save_weights/mix_0.1_uda.pth'
    model_a.load_state_dict(torch.load(path_a, map_location='cpu')['model'])
    model_b.load_state_dict(torch.load(path_b, map_location='cpu')['student_model'])
    model_c.load_state_dict(torch.load(path_c, map_location='cpu')['student_model'])
    model_d.load_state_dict(torch.load(path_d, map_location='cpu')['student_model'])
    model_a.eval()
    model_b.eval()
    model_c.eval()
    model_d.eval()

    val_dataset_info = [{"dataset":"animal_pose","mode":"val"}]
    val_dataset = MixKeypoint(root='../../dataset',merge_info=val_dataset_info,transform=data_transform['val'])
    data_loader = DataLoader(val_dataset,
                             batch_size=1,
                             sampler=RandomSampler(val_dataset),
                             pin_memory=True,
                             num_workers=1,
                             drop_last=False,
                             collate_fn=val_dataset.collate_fn)
    with amp.autocast(enabled=True):
        with torch.no_grad():
            for imgs,targets in data_loader:
                logit_a = model_a(imgs)
                logit_b = model_b(imgs)
                logit_c = model_c(imgs)
                logit_d = model_d(imgs)
                # if neck keypoint is visible
                coord_gt,_ = get_max_preds(targets[0]['heatmap'].unsqueeze(0))
                coord_a, _ = get_max_preds(logit_a)
                coord_b, _ = get_max_preds(logit_b)
                coord_c, _ = get_max_preds(logit_c)
                coord_d, _ = get_max_preds(logit_d)
                coord_gt *= 4
                coord_a *= 4
                coord_b *= 4
                coord_c *= 4
                coord_d *= 4
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

                exclusive_kp_index_a = [8]
                exclusive_kp_index_b = [2, 3, 6, 7]

                coord_gt = coord_gt[:,exclusive_kp_index_b,:]
                coord_a = coord_a[:,exclusive_kp_index_b,:]
                coord_b = coord_b[:,exclusive_kp_index_b,:]
                coord_c = coord_c[:,exclusive_kp_index_b,:]
                coord_d = coord_d[:,exclusive_kp_index_b,:]

                images_tensor = imgs * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
                images_np = images_tensor.numpy()
                from matplotlib import pyplot as plt
                plt.imshow(images_np[0].transpose(1, 2, 0))
                plt.scatter(coord_gt[0,:,0], coord_gt[0,:,1], c='white', marker='o',label=f'GT ',s=60)
                plt.scatter(coord_a[0,:,0], coord_a[0,:,1], c='black', marker='o',label=f'SL ',s=60)
                plt.scatter(coord_b[0,:,0], coord_b[0,:,1], c='green', marker='o',label=f'SSL ',s=60)
                plt.scatter(coord_c[0,:,0], coord_c[0,:,1], c='blue', marker='o',label=f'SSL+self ',s=60)
                plt.scatter(coord_d[0,:,0], coord_d[0,:,1], c='red', marker='o',label=f'SSL+uda ',s=60)
                plt.title('Animal Pose Exclusive Keypoints')
                plt.legend()
                plt.show()


def different_loss_comparison():
    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    keypoints_path = f"../info/ap_10k_animal_pose_union_keypoints_format.json"
    with open(keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    fixed_size = (256, 256)
    heatmap_hw = (fixed_size[0] // 4, fixed_size[1] // 4)
    kps_weights = np.array(animal_kps_info["kps_weights"],
                           dtype=np.float32).reshape((21,))
    data_transform = {
        "val": transforms.Compose([
            transforms.LabelFormatTransAP10KAnimalPose(extend_flag=True),
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    data_root = "../../dataset"
    dataset_info = [{"dataset":"ap_10k","mode":"train"},{"dataset":"animal_pose","mode":"train"}]
    dataset = MixKeypoint(root=data_root, merge_info=dataset_info, transform=data_transform['val'],num_joints=21)
    # Ls / + Lu / +Lf
    weights_path_a = "../pretrained_weights/ls.pth"
    weights_path_b = "../pretrained_weights/ls_lu.pth"
    weights_path_c = "../pretrained_weights/ls_lu_lf.pth"
    model_a = HighResolutionNet(num_joints=21)
    model_b = HighResolutionNet(num_joints=21)
    model_c = HighResolutionNet(num_joints=21)
    model_a.load_state_dict(torch.load(weights_path_a)['state_dict'])
    model_b.load_state_dict(torch.load(weights_path_b))
    model_c.load_state_dict(torch.load(weights_path_c))

    model_a.eval()
    model_b.eval()
    model_c.eval()
    model_a.to("cuda:0")
    model_b.to("cuda:0")
    model_c.to("cuda:0")

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             pin_memory=True,
                             sampler=RandomSampler(dataset),
                             num_workers=1,
                             drop_last=False,
                             collate_fn=dataset.collate_fn)
    models = [model_a,model_b,model_c]
    exclusive_indices = [3,17,18,19,20]
    titles = ['Ls','Ls + Lu','Ls + Lu + Lf']
    with torch.no_grad():
        for imgs,targets in data_loader:
            img_path = targets[0]['image_path']
            base_name = os.path.basename(img_path).split(".")[0]
            if base_name != "000000023468":
                continue
            cur_img = mpimg.imread(img_path)
            imgs = imgs.to("cuda:0")

            fig,axs = plt.subplots(1,4,figsize=(20,5),num=os.path.basename(img_path).split('.')[0])
            for i,model in enumerate(models):
                # inference
                outputs = model(imgs)

                flipped_images = transforms.flip_images(imgs)
                flipped_outputs = model(flipped_images)
                flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
                flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
                outputs = (outputs + flipped_outputs) * 0.5
                reverse_trans = [t["reverse_trans"] for t in targets]
                outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
                # outputs = transforms.get_final_preds(outputs.detach().cpu(), reverse_trans, post_processing=True)
                vis = np.reshape(outputs[1],(21,))
                kps = np.reshape(outputs[0],(21,2))
                axs[i].imshow(cur_img)
                for j in range(len(vis)):
                    alpha = 1
                    if vis[j] > 0.4:
                        color = 'white'
                    elif vis[j] > 0.1:
                        color = 'red'
                    else:
                        continue
                    if j in exclusive_indices:
                        marker = 'o'
                        if vis[j]:
                            color = 'green'
                        size = 70
                    else:
                        marker = 'o'
                        size = 50
                    axs[i].scatter(*kps[j],s=size,alpha=alpha,c=color,marker=marker,edgecolors='black')

                axs[i].set_title(titles[i])
                axs[i].axis('off')
                draw_line(axs[i],kps,vis)
            # Draw GT
            gt_coords = targets[0]['keypoints_ori']
            gt_vis = targets[0]['visible_ori']
            for j in range(len(gt_vis)):
                if gt_vis[j] > 0:
                    if j in exclusive_indices:
                        size = 70
                        color = 'green'
                    else:
                        size = 50
                        color = 'white'
                    axs[3].scatter(*gt_coords[j], s=size,c=color,edgecolors='black')
            axs[3].set_title('GT')
            axs[3].imshow(cur_img)
            axs[3].axis('off')
            draw_line(axs[3],gt_coords,gt_vis)
            plt.subplots_adjust(wspace=0.02,hspace=0.02,left=0.02,right=0.98)
            plt.show()


def draw_line(ax,coords,vis):
    info_path = '../info/ap_10k_animal_pose_union_keypoints_format.json'
    with open(info_path,'r') as f:
        skeletons = json.load(f)['skeleton_o']
    for pair in skeletons:
        if vis[pair[0]] > 0.1 and vis[pair[1]] > 0.1:
            ax.plot([coords[pair[0]][0],coords[pair[1]][0]],[coords[pair[0]][1],coords[pair[1]][1]],c='yellow')


if __name__ == '__main__':
    # seed = 2
    # set_seed(seed)
    # dataset = 'ap_10k'
    # num_joints = 17
    # dataset = 'animal_pose'
    # num_joints = 20
    # dataset = 'tigdog'
    # num_joints = 19
    # gt_show(dataset,num_joints)
    # dt_show()'
    different_loss_comparison()
