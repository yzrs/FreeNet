import os.path
import random
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from train_utils import transforms
from train_utils.dataset import CocoKeypoint, MixKeypoint
from models.hrnet import HighResolutionNet
from train_utils.transforms import get_max_preds


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dt_gt_comparison():
    set_seed(3)
    keypoints_path = f"../info/union_definition.json"
    with open(keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    fixed_size = (256, 256)
    heatmap_hw = (fixed_size[0] // 4, fixed_size[1] // 4)
    kps_weights = np.array(animal_kps_info["kps_weights"],
                           dtype=np.float32).reshape((24,))
    data_transform = {
        "val": transforms.Compose([
            transforms.LabelFormatTransUnion(extend_flag=True),
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    data_root = "../../dataset"
    # dataset_info = [{"dataset":"ap_10k","mode":"train"},{"dataset":"animal_pose","mode":"train"},
    #                 {"dataset":"tigdog","mode":"train"}]
    # dataset_info = [{"dataset":"ap_10k","mode":"train"}]
    # dataset_info = [{"dataset":"animal_pose","mode":"train"}]
    dataset_info = [{"dataset":"tigdog","mode":"train"}]
    dataset = MixKeypoint(root=data_root, merge_info=dataset_info, transform=data_transform['val'],num_joints=24)
    # Ls / + Lu / +Lf
    weights_path = "../saved_weights/union_ours_oks_54.7.pth"
    model = HighResolutionNet(num_joints=24)
    checkpoint = torch.load(weights_path)
    load_flag = False
    for checkpoint_key in ['student_model','model','state']:
        if checkpoint_key in checkpoint:
            model.load_state_dict(checkpoint[checkpoint_key])
            load_flag = True
            break
    if not load_flag:
        model.load_state_dict(checkpoint)
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
        "L_eye","R_eye","nose","neck","tail",
        "L_F_hip","L_F_knee","L_F_paw",
        "R_F_hip","R_F_knee","R_F_paw",
        "L_B_hip","L_B_knee","L_B_paw",
        "R_B_hip","R_B_knee","R_B_paw",
        "L_ear","R_ear","throat","wither",
        "chin","L_shoulder","R_shoulder",
    """

    catIds = [key for key in dataset.coco_lists[0]['coco'].catToImgs if key != 0]
    # [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    # 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53]
    with torch.no_grad():
        for imgs,targets in data_loader:
            # if targets[0]['dataset'] != 'ap_10k':
            #     continue
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
            outputs_pose = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            dt_vis = np.reshape(outputs_pose[1], (24,))
            dt_coords = np.reshape(outputs_pose[0], (24,2))
            # Draw GT
            gt_coords = targets[0]['keypoints_ori']
            gt_vis = targets[0]['visible']

            ap10_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            animalpose_index = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            tigdog_index = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
            ap10_animalpose_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            union_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
            mix_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,22,23]

            head_index = [0,1,2,3,17,18,19,21]
            front_index = [5,6,7,8,9,10,20,22,23]
            back_index = [4,11,12,13,14,15,16]

            # titles = ['AP-10K','Animal Pose','Tigdog','AP-10K + Animal Pose','AP-10K + Animal Pose + Tigdog','GT']
            titles = ['Prediction','GT']

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

            # kps_indices = [ap10_index,animalpose_index,tigdog_index,ap10_animalpose_index,mix_index,gt_index]
            kps_indices = [mix_index,gt_index]

            # fig,axs = plt.subplots(1,6,figsize=(20,5),num=os.path.basename(img_path).split('.')[0])
            fig,axs = plt.subplots(1,2,figsize=(20,10),num=os.path.basename(img_path).split('.')[0])
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
                    axs[i].scatter(*cur_coords[j],s=100,c=color,edgecolors='black')

                axs[i].set_title(titles[i],fontsize=16)
                axs[i].axis('off')

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
                    axs[len(kps_indices)-1].scatter(*gt_coords[j], s=100,c=color,edgecolors='black')
            axs[len(kps_indices)-1].set_title('AP-10K GT',fontsize=16)
            axs[len(kps_indices)-1].imshow(cur_img)
            axs[len(kps_indices)-1].axis('off')
            plt.subplots_adjust(wspace=0.01,hspace=0.02,left=0.02,right=0.98)
            plt.show()


def different_joints_comparision(seed):
    ap10k_targets_img_names = []
    animalpose_targets_img_names = []
    tigdog_targets_img_names = []
    ap10k_targets_img_names.append('000000002118.jpg')
    ap10k_targets_img_names.append('000000019309.jpg')
    ap10k_targets_img_names.append('000000020851.jpg')
    ap10k_targets_img_names.append('000000055842.jpg')
    animalpose_targets_img_names.append('2010_001994.jpg')
    animalpose_targets_img_names.append('2011_000385.jpg')
    animalpose_targets_img_names.append('co183.jpeg')
    animalpose_targets_img_names.append('ho40.jpeg')
    tigdog_targets_img_names.append('00047399.jpg')

    dataset = "ap_10k"
    mode = "test"
    targets_img_names = ap10k_targets_img_names
    threshold = 0.15
    size = 50

    set_seed(seed)
    keypoints_path = f"../info/union_definition.json"
    with open(keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    fixed_size = (256, 256)
    heatmap_hw = (fixed_size[0] // 4, fixed_size[1] // 4)
    kps_weights = np.array(animal_kps_info["kps_weights"],
                           dtype=np.float32).reshape((24,))
    data_transform = {
        "val": transforms.Compose([
            transforms.LabelFormatTransUnion(extend_flag=True),
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    data_root = "../../dataset"
    # dataset_info = [{"dataset":"ap_10k","mode":"train"},{"dataset":"animal_pose","mode":"train"},
    #                 {"dataset":"tigdog","mode":"train"}]
    # dataset_info = [{"dataset":"ap_10k","mode":"train"}]
    # dataset_info = [{"dataset":"animal_pose","mode":"train"}]
    # dataset_info = [{"dataset":"tigdog","mode":"train"}]
    dataset_info = [{"dataset":dataset,"mode":mode}]
    dataset = MixKeypoint(root=data_root, merge_info=dataset_info, transform=data_transform['val'],num_joints=24)

    ap10k_weights_path = "../saved_weights/random_0.1_ap10k.pth"
    ap10k_model = get_model(ap10k_weights_path,num_joints=17,device="cuda:0")
    tigdog_weights_path = "../saved_weights/random_0.1_tigdog.pth"
    tigdog_model = get_model(tigdog_weights_path,num_joints=19,device="cuda:0")
    animalpose_weights_path = "../saved_weights/random_0.1_animalpose.pth"
    animalpose_model = get_model(animalpose_weights_path,num_joints=20,device="cuda:0")
    ap10k_animalpose_weights_path = "../saved_weights/ours_0.1_ap10k_animalpose.pth"
    ap10k_animalpose_model = get_model(ap10k_animalpose_weights_path,num_joints=21,device="cuda:0")
    union_weights_path = "../saved_weights/union_ours_oks_54.7.pth"
    union_model = get_model(union_weights_path,num_joints=24,device="cuda:0")

    models = [ap10k_model,tigdog_model,animalpose_model,ap10k_animalpose_model,union_model]

    flip_pairs = []
    ap10k_flip_pairs = [[0, 1], [5, 8], [6, 9], [7, 10], [11, 14], [12, 15], [13, 16]]
    animalpose_flip_pairs = [[0,1], [3,4], [5,6], [7,8],[9,10],[11,12],[13,14],[15,16]]
    tigdog_flip_pairs = [[0,1], [3,4],[5,6],[8,9],[10,11],[12,13],[14,15],[16,17]]
    ap10k_animalpose_flip_pairs = [[0, 1], [5, 8], [6, 9], [7, 10], [11, 14], [12, 15], [13, 16], [17,18]]
    union_flip_pairs = [[0,1], [5,8], [6,9], [7,10],[11,14],[12,15],[13,16],[17,18],[22,23]]
    flip_pairs.append(ap10k_flip_pairs)
    flip_pairs.append(tigdog_flip_pairs)
    flip_pairs.append(animalpose_flip_pairs)
    flip_pairs.append(ap10k_animalpose_flip_pairs)
    flip_pairs.append(union_flip_pairs)

    convert_mapping = []
    ap10k2union_mapping = [[0,0],[1,1],[2,2],[3,3],[4,4],
                    [5,5],[6,6],[7,7],[8,8],
                    [9,9],[10,10],[11,11],[12,12],
                    [13,13],[14,14],[15,15],[16,16]]
    animalpose2union_mapping = [[0,0],[1,1],[2,2],[3,17],[4,18],
                         [5,5],[6,8],[7,11],[8,14],
                         [9,6],[10,9],[11,12],[12,15],
                         [13,7],[14,10],[15,13],[16,16],
                         [17,19],[18,20],[19,4]]
    tigdog2union_mapping = [[0,0],[1,1],[2,21],
                    [3,7],[4,10],[5,13],[6,16],
                    [7,4],
                    [8,6],[9,9],[10,12],[11,15],
                    [12,22],[13,23],
                    [14,5],[15,8],[16,11],[17,14],
                    [18,3]]
    ap10k_animalpose2union_mapping = [[0,0],[1,1],[2,2],[3,3],[4,4],
                                [5,5],[6,6],[7,7],[8,8],
                                [9,9],[10,10],[11,11],[12,12],
                                [13,13],[14,14],[15,15],[16,16],
                                [17,17],[18,18],[19,19],[20,20]]
    union2union_mapping = [[0,0],[1,1],[2,2],[3,3],[4,4],
                        [5,5],[6,6],[7,7],[8,8],
                        [9,9],[10,10],[11,11],[12,12],
                        [13,13],[14,14],[15,15],[16,16],
                        [17,17],[18,18],[19,19],[20,20],
                       [21,21],[22,22],[23,23]]

    convert_mapping.append(ap10k2union_mapping)
    convert_mapping.append(tigdog2union_mapping)
    convert_mapping.append(animalpose2union_mapping)
    convert_mapping.append(ap10k_animalpose2union_mapping)
    convert_mapping.append(union2union_mapping)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             pin_memory=True,
                             sampler=RandomSampler(dataset),
                             num_workers=1,
                             drop_last=False,
                             collate_fn=dataset.collate_fn)

    """
        "L_eye","R_eye","nose","neck","tail",
        "L_F_hip","L_F_knee","L_F_paw",
        "R_F_hip","R_F_knee","R_F_paw",
        "L_B_hip","L_B_knee","L_B_paw",
        "R_B_hip","R_B_knee","R_B_paw",
        "L_ear","R_ear","throat","wither",
        "chin","L_shoulder","R_shoulder",
    """

    # catIds = [key for key in dataset.coco_lists[0]['coco'].catToImgs if key != 0]
    with torch.no_grad():
        for imgs,targets in tqdm(data_loader):
            img_path = targets[0]['image_path']
            if os.path.basename(img_path) not in targets_img_names:
                continue
            cur_img = mpimg.imread(img_path)
            imgs = imgs.to("cuda:0")

            all_dt_vis = []
            all_dt_coords = []

            for i,model in enumerate(models):
                dt_vis,dt_coord = get_prediction(model, imgs, flip_pairs[i],targets,convert_mapping[i])
                all_dt_vis.append(dt_vis)
                all_dt_coords.append(dt_coord)

            ap10_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            animalpose_index = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            tigdog_index = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
            ap10_animalpose_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            union_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

            head_index = [0,1,2,3,17,18,19,21]
            front_index = [5,6,7,8,9,10,20,22,23]
            back_index = [4,11,12,13,14,15,16]

            plt.rcParams['figure.subplot.left'] = 0
            plt.rcParams['figure.subplot.right'] = 1
            plt.rcParams['figure.subplot.wspace'] = 0.01

            kps_indices = [ap10_index,tigdog_index,animalpose_index,ap10_animalpose_index,union_index]
            fig,axs = plt.subplots(1,len(kps_indices),figsize=(10,10))
            for i in range(len(kps_indices)):
                cur_index = kps_indices[i]
                cur_coords = all_dt_coords[i][cur_index]
                cur_vis = all_dt_vis[i][cur_index]

                axs[i].imshow(cur_img)
                for j in range(len(cur_vis)):
                    if cur_vis[j] > threshold:
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
                    axs[i].scatter(*cur_coords[j],s=size,c=color,edgecolors='black')
                axs[i].axis('off')

            plt.show()


def get_model(path,num_joints,device):
    model = HighResolutionNet(num_joints=num_joints)
    checkpoint = torch.load(path)
    load_flag = False
    for checkpoint_key in ['student_model','state_dict','model']:
        if checkpoint_key in checkpoint:
            model.load_state_dict(checkpoint[checkpoint_key])
            load_flag = True
            print("Loaded model from {}".format(checkpoint_key))
            break
    if not load_flag:
        model.load_state_dict(checkpoint)
        print("Loaded model directly")
    model.eval()
    model.to(device)
    return model


def get_prediction(model,imgs,flip_pairs,targets,mapping):
    # inference
    outputs = model(imgs)
    dt_joints_num = outputs.shape[1]
    flipped_images = transforms.flip_images(imgs)
    flipped_outputs = model(flipped_images)
    flipped_outputs = transforms.flip_back(flipped_outputs, flip_pairs)
    flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
    outputs = (outputs + flipped_outputs) * 0.5
    reverse_trans = [t["reverse_trans"] for t in targets]
    outputs_pose = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
    dt_vis = np.reshape(outputs_pose[1], (dt_joints_num,))
    dt_coords = np.reshape(outputs_pose[0], (dt_joints_num, 2))
    # convert to 24 joints prediction
    convert_vis = np.zeros(24)
    convert_coord = np.zeros((24,2))

    for pair in mapping:
        convert_coord[pair[1]] = dt_coords[pair[0]]
        convert_vis[pair[1]] = dt_vis[pair[0]]

    return convert_vis,convert_coord


if __name__ == '__main__':
    different_joints_comparision(seed=1)
