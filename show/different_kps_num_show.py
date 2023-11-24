import os.path
import random
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from train_utils import transforms
from train_utils.dataset import MixKeypoint
from models.hrnet import HighResolutionNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def different_kp_num_comparison():
    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    keypoints_path = f"../info/keypoints_definition.json"
    with open(keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    fixed_size = (256, 256)
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
    data_root = "../../dataset"
    dataset_info = [{"dataset":"ap_10k","mode":"train"},{"dataset":"animal_pose","mode":"train"},
                    {"dataset":"tigdog","mode":"train"}]
    dataset = MixKeypoint(root=data_root, merge_info=dataset_info, transform=data_transform['val'],num_joints=26)
    # Ls / + Lu / +Lf
    weights_path = "../pretrained_weights/mix_26_SL.pth"
    model = HighResolutionNet(num_joints=26)
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to("cuda:0")

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             pin_memory=True,
                             sampler=RandomSampler(dataset),
                             num_workers=1,
                             drop_last=False,
                             collate_fn=dataset.collate_fn)

    """
        "L_eye","R_eye","L_ear","R_ear","nose",
        "chin","throat","wither","neck","L_shoulder",
        "R_shoulder","L_F_hip","R_F_hip","L_B_hip","R_B_hip",
        "L_F_knee","R_F_knee","L_B_knee","R_B_knee","L_F_ankle",
        "R_F_ankle","L_F_paw","R_F_paw","L_B_paw","R_B_paw",
        "tail"
    """

    with torch.no_grad():
        for imgs,targets in data_loader:
            if targets[0]['dataset'] != 'tigdog':
                continue
            img_path = targets[0]['image_path']
            cur_img = mpimg.imread(img_path)
            imgs = imgs.to("cuda:0")

            # inference
            outputs = model(imgs)
            flipped_images = transforms.flip_images(imgs)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, animal_kps_info["flip_pairs"])
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            dt_vis = np.reshape(outputs[1], (26,))
            dt_coords = np.reshape(outputs[0], (26,2))
            # Draw GT
            gt_coords = targets[0]['keypoints_ori']
            gt_vis = targets[0]['visible']

            ap10_index = [0,1,4,8,11,12,13,14,15,16,17,18,21,22,23,24,25]
            animalpose_index = [0,1,2,3,4,6,7,11,12,13,14,15,16,17,18,21,22,23,24,25]
            tigdog_index = [0,1,5,8,9,10,11,12,13,14,15,16,17,18,21,22,23,24,25]
            ap10_animalpose_index = [0,1,2,3,4,6,7,8,11,12,13,14,15,16,17,18,21,22,23,24,25]
            mix_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,22,23,24,25]

            exclusive_indices = [2,3,4,5,6,7,8,9,10]

            head_index = [0,1,2,3,4,5,6,7,8]
            front_index = [9,10,11,12,15,16,21,22]
            back_index = [13,14,17,18,23,24,25]

            titles = ['AP-10K','Animal Pose','Tigdog','AP-10K + Animal Pose','AP-10K + Animal Pose + Tigdog','GT']

            if targets[0]['dataset'] == 'ap_10k':
                gt_index = ap10_index
                gt_vis = gt_vis[ap10_index]
                gt_coords = gt_coords[ap10_index]
            elif targets[0]['dataset'] == 'animal_pose':
                gt_index = animalpose_index
                gt_vis = gt_vis[animalpose_index]
                gt_coords = gt_coords[animalpose_index]
            else:
                gt_index = tigdog_index
                gt_vis = gt_vis[tigdog_index]
                gt_coords = gt_coords[tigdog_index]

            kps_indices = [ap10_index,animalpose_index,tigdog_index,ap10_animalpose_index,mix_index,gt_index]

            fig,axs = plt.subplots(1,6,figsize=(20,5),num=os.path.basename(img_path).split('.')[0])
            for i in range(len(kps_indices)-1):
                cur_index = kps_indices[i]
                cur_coords = dt_coords[cur_index]
                cur_vis = dt_vis[cur_index]

                axs[i].imshow(cur_img)
                for j in range(len(cur_vis)):
                    if cur_vis[j] > 0.3:
                        # color = 'white'
                        # if cur_index[j] in exclusive_indices:
                        #     color = 'red'
                        if cur_index[j] in head_index:
                            color = 'white'
                        elif cur_index[j] in front_index:
                            color = 'red'
                        elif cur_index[j] in back_index:
                            color = 'yellow'
                        else:
                            continue
                    else:
                        continue
                    axs[i].scatter(*cur_coords[j],c=color,edgecolors='black')

                axs[i].set_title(titles[i])
                axs[i].axis('off')
                # draw_line(axs[i],kps,vis)

            for j in range(len(gt_vis)):
                if gt_vis[j] > 0:
                    color = 'white'
                    # if gt_index[j] in exclusive_indices:
                    #     color = 'red'
                    if gt_index[j] in head_index:
                        color = 'white'
                    elif gt_index[j] in front_index:
                        color = 'red'
                    elif gt_index[j] in back_index:
                        color = 'yellow'
                    axs[len(kps_indices)-1].scatter(*gt_coords[j], c=color,edgecolors='black')
            axs[len(kps_indices)-1].set_title('GT')
            axs[len(kps_indices)-1].imshow(cur_img)
            axs[len(kps_indices)-1].axis('off')
            # draw_line(axs[3],gt_coords,gt_vis)
            plt.subplots_adjust(wspace=0.01,hspace=0.02,left=0.02,right=0.98)
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
    different_kp_num_comparison()
