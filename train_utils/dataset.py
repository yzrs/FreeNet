import argparse
import datetime
import os
import copy
import torch
import numpy as np
import cv2
import random
import torch.utils.data as data
from pycocotools.coco import COCO
from train_utils.transforms import OriginalLabelFormatTrans, NoLabelFormatTrans, OriginalLabelFormatTransAP10KAnimalPose
from train_utils import transforms
import json
import logging

logger = logging.getLogger(__name__)


class CocoKeypoint(data.Dataset):
    def __init__(self,
                 root,
                 dataset,
                 mode,
                 transform=None,
                 fixed_size=(256, 256),
                 data_type=None,
                 num_joints=17
                 ):
        super().__init__()
        assert dataset in ["ap_10k", "ap_60k", "animal_pose", "tigdog", "tigdog_horse", "tigdog_tiger"]
        assert mode in ["train", "val", "test"]
        self.num_joints = num_joints
        if dataset == "ap_10k":
            anno_file = f"ap10k-{mode}-split1.json"
        elif dataset == "animal_pose":
            anno_file = f"animal_pose_{mode}.json"
            self.num_joints = 20
        elif dataset == "tigdog":
            anno_file = f"tigdog_{mode}.json"
            self.num_joints = 19
        else:
            anno_file = f"{dataset}_{mode}.json"
            self.num_joints = 19
        self.dataset = dataset
        self.mode = mode
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_root = os.path.join(root, "data")
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        self.anno_path = os.path.join(root, "annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)
        self.data_type = data_type
        assert data_type in ["keypoints", "bbox_only", "no_label", "blank"]

        self.fixed_size = fixed_size
        self.transforms = transform
        self.coco = COCO(self.anno_path)
        img_ids = list(sorted(self.coco.imgs.keys()))

        # label / only bounding box / without any label
        self.imgs_with_keypoints = []
        self.imgs_with_bbox_only = []
        self.imgs_without_labels = []

        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                if "bbox" in ann and ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
                    info = {
                        "anno_id": ann['id'],
                        "image_path": os.path.join(self.img_root, img_info["file_name"]),
                        "image_id": img_id,
                        "image_width": img_info['width'],
                        "image_height": img_info['height'],
                        "category_id": ann['category_id'],
                        "box": ann['bbox'],
                        "score": ann['score'] if "score" in ann else 1,
                        "obj_origin_hw": [ann['bbox'][3], ann['bbox'][2]],
                        "obj_index": 0,
                    }
                    if "keypoints" in ann and max(ann["keypoints"]) > 0:
                        keypoints = np.array(ann["keypoints"]).reshape([-1, 3])
                        visible = keypoints[:, 2]
                        keypoints = keypoints[:, :2]
                        info["keypoints"] = keypoints
                        info["visible"] = visible
                        info['obj_index'] = len(self.imgs_with_keypoints)
                        info['mode'] = "label"
                        self.imgs_with_keypoints.append(info)

                    else:
                        info["keypoints"] = np.zeros((self.num_joints, 2))
                        info["visible"] = np.zeros(self.num_joints)
                        info['obj_index'] = len(self.imgs_with_bbox_only)
                        info['mode'] = "unlabel"
                        self.imgs_with_bbox_only.append(info)
            if len(anns) == 0:
                info = {
                    "anno_id": len(self.imgs_without_labels),
                    "image_path": os.path.join(self.img_root, img_info["file_name"]),
                    "image_id": img_id,
                    "image_width": img_info['width'],
                    "image_height": img_info['height'],
                    "category_id": 0,
                    "box": [0, 0, img_info['width'], img_info['height']],
                    "score": 1,
                    "obj_origin_hw": [img_info['height'], img_info['width']],
                    "obj_index": len(self.imgs_without_labels),
                    "keypoints": np.zeros((self.num_joints, 2)),
                    "visible": np.zeros(self.num_joints),
                    "mode": "unlabel"
                }
                self.imgs_without_labels.append(info)

        self.valid_list = []
        if self.data_type == "keypoints":
            self.valid_list = copy.copy(self.imgs_with_keypoints)
        elif self.data_type == "bbox_only":
            self.valid_list = copy.copy(self.imgs_with_bbox_only)
        elif self.data_type == "no_label":
            self.valid_list = copy.copy(self.imgs_without_labels)
        else:
            pass

        del self.imgs_with_keypoints
        del self.imgs_with_bbox_only
        del self.imgs_without_labels

    def __getitem__(self, index):
        # 根据自定义的索引顺序读取数据
        target = copy.deepcopy(self.valid_list[index])

        image = cv2.imread(target["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.valid_list)

    def sample_few(self, args, num_lamda=0.1):
        num = int(len(self.valid_list) * num_lamda)
        val_list = random.sample(self.valid_list, num)
        self.valid_list = val_list
        self.get_kps_num(args, num_lamda)

    def get_kps_num(self, args):
        # now = datetime.datetime.now()
        # now_time = now.strftime('%Y-%m_%d_%H-%M-%S')
        # 统计每个点对应的数量，并保存其信息到本地的statistics文件中
        # write_path = f"./statistics_file/num_results/{self.dataset}_{self.mode}_{lamda}_kp_num_{now_time}.json"
        kp_num = [0 for _ in range(self.num_joints)]
        for label in self.valid_list:
            vis = label['visible']
            for val_index, val in enumerate(vis):
                if val > 0:
                    kp_num[val_index] += 1
        with open(args.keypoints_path) as f:
            kps = json.load(f)['keypoints']
        res = {key: val for key, val in zip(kps, kp_num)}
        logger.info("sample kps num :{}".format(res))
        # with open(write_path, 'w') as f:
        #     json.dump(res, f, indent=4)

    # show the GT info of the given animal category
    def animal_gt_show(self, animal_category, shuffle=False):
        coco = self.coco

        # create a resizable window
        cv2.namedWindow('GT Keypoints', cv2.WINDOW_NORMAL)
        # maximize the window
        # cv2.setWindowProperty('GT Keypoints', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        ids_list = coco.getImgIds()
        if shuffle:
            random.shuffle(ids_list)
        for img_id in ids_list:
            # get the image
            img = coco.loadImgs(img_id)[0]

            # get the GT keypoints
            gt_ann_ids = coco.getAnnIds(imgIds=img['id'])
            gt_anns = coco.loadAnns(gt_ann_ids)

            # animal category info
            category = gt_anns[0]['category_id']
            cat = coco.loadCats(category)[0]
            if animal_category is not None and cat['name'] != animal_category:
                continue

            # load the image using OpenCV
            img_path = os.path.join(self.img_root, img['file_name'])
            gt_image = cv2.imread(img_path)

            # draw keypoints in GT image
            for gt_ann in gt_anns:

                gt = gt_ann['keypoints']
                gt = np.array(gt)
                gt_x = gt[0::3]
                gt_y = gt[1::3]
                gt_vis = gt[2::3]

                # iterate over keypoints and draw them based on visibility
                for i in range(len(gt_x)):
                    color = (255, 0, 0)
                    size = 3
                    if gt_vis[i] > 0:  # if the keypoint is visible
                        # if i == 3 or i == 4 or i == 5 or i == 6:
                        # if i == 18:
                        #     color = (0, 0, 255)
                        #     size = 6
                        cv2.circle(gt_image, (int(gt_x[i]), int(gt_y[i])), size, color, -1)

            # display the GT image with keypoints
            cv2.imshow('GT Keypoints', gt_image)
            # Wait for a key press to exit
            key = cv2.waitKey(0)
            # Close the window if press q
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord('c'):
                continue
            elif key == ord('s'):
                # Save the image
                # cv2.imwrite(f"{self.dataset}_{img['file_name']}", gt_image)
                cv2.imwrite(f"{self.dataset}_{img['file_name'].split('/')[1]}", gt_image)
                print(f"{self.dataset}_{img['file_name']} done")
            elif key == ord('p'):
                # Pause and wait for another key press
                cv2.waitKey(0)

    # 输入需要屏蔽的关键点对应的索引
    def block_specific_kp(self, args, kp_index):
        if len(kp_index) > 0:
            for ann in self.valid_list:
                for ind in kp_index:
                    ann['visible'][ind] = 0
        self.get_kps_num(args, 1)
        # print("debug")

    def get_id_set(self, id_num):
        anns = random.sample(self.valid_list, id_num)
        id_set = set()
        for ann in anns:
            id_set.add(ann['anno_id'])
        return id_set

    def sample_anns_by_id(self, id_set):
        current_anns = [ann for ann in self.valid_list if ann['anno_id'] in id_set]
        self.valid_list = current_anns

    def filter_anns_by_id(self, id_set):
        current_anns = [ann for ann in self.valid_list if ann['anno_id'] not in id_set]
        self.valid_list = current_anns

    def sample_anns_by_num(self, num):
        id_set = self.get_id_set(num)
        self.sample_anns_by_id(id_set)

    def sample_anns_by_imgs_ids_file(self,path):
        with open(path,'r') as f:
            ids = json.load(f)
        anns_ids_set = set()
        for img_id in ids:
            img = self.coco.loadImgs(img_id)[0]
            anns_ids = self.coco.getAnnIds(imgIds=img['id'],iscrowd=None)
            anns = self.coco.loadAnns(anns_ids)
            for ann in anns:
                anns_ids_set.add(ann['id'])
        self.sample_anns_by_id(anns_ids_set)

    # 只是根据点的数量去选取，没有考虑动物种类的问题
    def get_qualified_few_shot(self,num,num_lambda=0.8):
        ids_set_path = f"info/{self.dataset}_few_shot_id_path_{num}.json"
        if not os.path.exists(ids_set_path):
            valid_ids_set = set()
            valid_ids_list = []
            for ann in self.valid_list:
                if np.count_nonzero(ann['visible']) > ann['visible'].shape[0] * num_lambda:
                    valid_ids_set.add(ann['anno_id'])
                    valid_ids_list.append(ann['anno_id'])

            ids_info = {"ids":valid_ids_list}
            with open(ids_set_path,'w') as f:
                json.dump(ids_info,f,indent=4)
        else:
            with open(ids_set_path,'r') as f:
                valid_ids_list = json.load(f)['ids']
            valid_ids_set = set()
            for val in valid_ids_list:
                valid_ids_set.add(val)

        self.valid_list = [ann for ann in self.valid_list if ann['anno_id'] in valid_ids_set]

    # 根据点的数量和动物的种类去选取few_shot
    def get_qualified_balanced_few_shot(self,args,num_per_species):
        ids_set_path = f"info/{self.dataset}_balanced_few_shot_id_path_{num_per_species}.json"
        if not os.path.exists(ids_set_path):
            anns_list = copy.copy(self.valid_list)
            valid_ids_set = set()
            valid_ids_list = [[] for _ in range(len(self.coco.cats))]
            sorted_anns_list = sorted(anns_list,key=lambda x:np.count_nonzero(x['visible']),reverse=True)
            for ann in sorted_anns_list:
                category_id = ann['category_id']
                if len(valid_ids_list[category_id-1]) < num_per_species:
                    valid_ids_list[category_id-1].append(ann['anno_id'])
                    valid_ids_set.add(ann['anno_id'])
                else:
                    continue

            ids_info = {"ids":valid_ids_list}
            with open(ids_set_path,'w') as f:
                json.dump(ids_info,f,indent=4)
        else:
            with open(ids_set_path,'r') as f:
                valid_ids_list = json.load(f)['ids']
            valid_ids_set = set()
            for val_list in valid_ids_list:
                for val in val_list:
                    valid_ids_set.add(val)

        self.valid_list = [ann for ann in self.valid_list if ann['anno_id'] in valid_ids_set]
        self.get_kps_num(args)

    def animal_category_num(self):
        category_num = len(self.coco.cats)
        animal_names = []
        animal_nums = [0] * category_num
        for cat in self.coco.cats:
            animal_names.append(self.coco.cats[cat]['name'])
        for ann in self.valid_list:
            animal_nums[ann['category_id']-1] += 1
        animal_num_dict = {name:num for name,num in zip(animal_names,animal_nums)}
        print(animal_num_dict)
        return animal_num_dict

    def animal_avg_keypoint_num(self):
        category_num = len(self.coco.cats)
        animal_nums = [[] for _ in range(category_num)]
        for ann in self.valid_list:
            animal_nums[ann['category_id'] - 1].append(np.count_nonzero(ann['visible']))
        print('debug')

    def get_avg_imgs_from_animals(self,avg_num):
        animals_dict = self.coco.catToImgs
        imgs_ids = [[] for _ in range(len(self.coco.cats))]
        for animal_category_id in animals_dict:
            ids = animals_dict[animal_category_id]
            if len(ids) > 0:
                tmp_ls = random.sample(ids,avg_num)
                imgs_ids[animal_category_id-1].extend(tmp_ls)
        anns_ids = set()
        for ls in imgs_ids:
            for img_id in ls:
                anns_ls = self.coco.imgToAnns[img_id]
                for ann in anns_ls:
                    anns_ids.add(ann['id'])
        self.sample_anns_by_id(anns_ids)

    def get_avg_animals(self,num):
        anns_ids = [[] for _ in range(len(self.coco.cats))]
        valid_ls = copy.copy(self.valid_list)
        random.shuffle(valid_ls)
        for ann in valid_ls:
            animal_id = ann['category_id']
            if len(anns_ids[animal_id-1]) < num:
                anns_ids[animal_id-1].append(ann['anno_id'])
        ids_set = set()
        for s_ls in anns_ids:
            for s_id in s_ls:
                ids_set.add(s_id)
        self.sample_anns_by_id(ids_set)

    def load_missing_anns(self,path):
        with open(path,'r') as f:
            anns_info = json.load(f)
        #  建立一个索引字典 去找img_id
        imgId2annIndex = {}
        for i,ann in enumerate(self.valid_list):
            img_id = ann['image_id']
            if img_id not in imgId2annIndex:
                imgId2annIndex[img_id] = [i]
            else:
                imgId2annIndex[img_id].append(i)
        # cnt = 0
        for i,ann in enumerate(anns_info):
            box = [val for val in ann['box']]
            related_anns_indices = imgId2annIndex[ann['image_id']]
            for ind in related_anns_indices:
                cur_box = [val for val in self.valid_list[ind]['box']]
                if self.compareList(box,cur_box):
                    # cnt += 1
                    # logger.info("{}_th ann updated {}/9122".format(i,cnt))
                    self.valid_list[ind]['visible'] = np.array(ann['vis'])
                    break

    @staticmethod
    def compareList(list_a,list_b):
        if len(list_a) != len(list_b):
            return False
        for i in range(len(list_a)):
            if abs(list_a[i]-list_b[i]) > 2:
                return False
        return True

    def set_transform(self,transform):
        self.transforms = transform

    @staticmethod
    def collate_fn(batch):
        imgs_tuple, targets_tuple = tuple(zip(*batch))
        imgs_tensor = torch.stack(imgs_tuple)
        return imgs_tensor, targets_tuple

    @staticmethod
    def collate_fn_mpl(batch):
        imgs_list, targets_list = list(zip(*batch))
        ori_imgs = [imgs[0] for imgs in imgs_list]
        aug_imgs = [imgs[1] for imgs in imgs_list]
        ori_imgs_tensor = torch.stack(ori_imgs)
        aug_imgs_tensor = torch.stack(aug_imgs)
        return (ori_imgs_tensor,aug_imgs_tensor), targets_list


# merge info is a list of dictionary. Like:
# [{"dataset":"ap_10k","mode":"test"}...}]
class MixKeypoint(data.Dataset):
    def __init__(self,
                 root,
                 merge_info,
                 transform=None,
                 fixed_size=(256, 256),
                 num_joints=26):
        super().__init__()
        self.root = root
        self.num_joints = num_joints
        self.length_list = []
        self.coco_lists = []
        self.valid_lists = []
        self.fixed_size = fixed_size
        self.transforms = transform
        self.anno_num = 0
        self.dataset_root = {
            "ap_10k": os.path.join(root, "ap_10k"),
            "animal_pose": os.path.join(root, "animal_pose"),
            "tigdog_horse": os.path.join(root, "tigdog_horse"),
            "tigdog_tiger": os.path.join(root, "tigdog_tiger"),
            "ap_10k_animal_pose_union":os.path.join(root,"merged_animal"),
            "tigdog": os.path.join(root, "tigdog"),
        }
        self.dataset_infos = merge_info
        for dataset_info in self.dataset_infos:
            dataset = dataset_info['dataset']
            mode = dataset_info['mode']
            assert dataset in ["ap_10k", "animal_pose", "tigdog","tigdog_horse", "tigdog_tiger","ap_10k_animal_pose_union"]
            if dataset == "ap_10k":
                dataset_info['num_joints'] = 17
            elif dataset == "animal_pose":
                dataset_info['num_joints'] = 20
            elif dataset == "tigdog_horse":
                dataset_info['num_joints'] = 19
            elif dataset == "tigdog_tiger":
                dataset_info['num_joints'] = 19
            elif dataset == "tigdog":
                dataset_info['num_joints'] = 19
            elif dataset == "ap_10k_animal_pose_union":
                dataset_info['num_joints'] = 21

            assert mode in ["train", "val", "test"]
            anno_path = os.path.join(self.dataset_root[dataset], f"annotations/{dataset}_{mode}.json")
            coco = COCO(anno_path)
            self.coco_lists.append({"coco": coco, "dataset": dataset, "mode": mode, "length": len(coco.anns)})

            tmp_list = []
            img_ids = list(sorted(coco.imgs.keys()))
            for img_id in img_ids:
                img_info = coco.loadImgs(img_id)[0]
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                for ann in anns:
                    keypoints = np.array(ann["keypoints"]).reshape([-1, 3])
                    visible = keypoints[:, 2]
                    # if np.count_nonzero(visible) == 0 :
                    if np.count_nonzero(keypoints) == 0 and np.count_nonzero(visible) == 0:
                        continue
                    keypoints = keypoints[:, :2]
                    info = {"anno_id": ann['id'],
                            "image_path": os.path.join(self.root, f"{dataset}/data", img_info["file_name"]),
                            "image_id": img_info['id'], "image_width": img_info['width'],
                            "image_height": img_info['height'], "category_id": ann['category_id'], "box": ann['bbox'],
                            "score": ann['score'] if "score" in ann else 1,
                            "obj_origin_hw": [ann['bbox'][3], ann['bbox'][2]], "obj_index": len(tmp_list),
                            "valid": True,
                            "keypoints": keypoints, "visible": visible, "keypoints_ori": keypoints,
                            "visible_ori": visible, "dataset": dataset, "mode": mode}
                    tmp_list.append(info)

            self.valid_lists.append(
                {"annotations": tmp_list, "dataset": dataset, "mode": mode, "length": len(tmp_list)})
            self.length_list.append(len(tmp_list))
            dataset_info['length'] = len(tmp_list)
            self.anno_num += len(tmp_list)

    def __getitem__(self, idx):
        for i, length in enumerate(self.length_list):
            if idx <= length - 1:
                target = copy.deepcopy(self.valid_lists[i]['annotations'][idx])
                image = cv2.imread(target["image_path"])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if self.transforms is not None:
                    image, target = self.transforms(image, target)
                return image, target
            else:
                idx -= length

    def __len__(self):
        total_length = 0
        for length in self.length_list:
            total_length += length
        return total_length

    # 从数据集中随机选择一定数量的注释作为新的数据集 7000/1000/500/500
    def get_valid_ann(self, args):
        selected_info_path = args.selected_info_path
        selected_lists = self.valid_lists.copy()
        if not os.path.exists(selected_info_path):
            # 将随机选择的ids保存在selected_info中
            # 更新dataset中相关信息
            selected_info = []
            for dataset_index, ls in enumerate(selected_lists):
                selected_num = 1
                if ls['dataset'] == 'ap_10k':
                    selected_num = 7000
                elif ls['dataset'] == 'animal_pose':
                    selected_num = 1000
                elif ls['dataset'] == 'tigdog_horse':
                    selected_num = 500
                elif ls['dataset'] == 'tigdog_tiger':
                    selected_num = 500
                anns = ls['annotations'].copy()
                anns = random.sample(anns, selected_num)
                valid_ids = []
                for single_info in anns:
                    valid_ids.append(single_info['anno_id'])
                ls['annotations'] = anns
                anno_num = len(anns)
                ls['length'] = anno_num
                tmp_dict = {"dataset": ls['dataset'], "mode": ls['mode'], "anno_ids": valid_ids, "anno_num": anno_num}
                selected_info.append(tmp_dict)

                # update info of the original dataset
                self.length_list[dataset_index] = anno_num
                self.dataset_infos[dataset_index]['length'] = anno_num
                self.anno_num = self.anno_num - ls['length'] + anno_num
                self.valid_lists[dataset_index] = ls
            with open(selected_info_path, 'w') as f:
                json.dump(selected_info, f, indent=4)
        else:
            with open(selected_info_path, 'r') as f:
                selected_info = json.load(f)
            for dataset_index, ls in enumerate(selected_lists):
                current_dataset = ls['dataset']
                current_mode = ls['mode']
                current_anns = ls['annotations'].copy()
                current_id_set = set()
                # generate its corresponding id set
                for info in selected_info:
                    if info['dataset'] == current_dataset and info['mode'] == current_mode:
                        ls['length'] = info['anno_num']
                        for one_id in info['anno_ids']:
                            if one_id not in current_id_set:
                                current_id_set.add(one_id)
                        break

                # 过滤注释列表
                current_anns = [ann for ann in current_anns if ann['anno_id'] in current_id_set]
                ls['annotations'] = current_anns

                anno_num = ls['length']
                # update the list to the original dataset
                self.valid_lists[dataset_index] = ls
                # update corresponding info
                self.anno_num = self.anno_num - self.length_list[dataset_index] + anno_num
                self.length_list[dataset_index] = anno_num
                self.dataset_infos[dataset_index]['length'] = anno_num

    # 必须在get_valid_ann之后使用
    def get_invalid_ann(self, args):
        selected_info_path = args.selected_info_path
        selected_lists = self.valid_lists.copy()

        with open(selected_info_path, 'r') as f:
            selected_info = json.load(f)
        for dataset_index, ls in enumerate(selected_lists):
            current_dataset = ls['dataset']
            current_mode = ls['mode']
            current_anns = ls['annotations'].copy()
            current_id_set = set()
            # generate its corresponding id set
            for info in selected_info:
                if info['dataset'] == current_dataset and info['mode'] == current_mode:
                    ls['length'] = ls['length'] - info['anno_num']
                    for one_id in info['anno_ids']:
                        if one_id not in current_id_set:
                            current_id_set.add(one_id)
                    break

            # 过滤注释列表
            current_anns = [ann for ann in current_anns if ann['anno_id'] not in current_id_set]
            for ann in current_anns:
                num_joint = ann['visible_ori'].shape[0]
                ann['visible'] = np.zeros((num_joint,), dtype=np.int64)
            ls['annotations'] = current_anns

            anno_num = ls['length']
            # update the list to the original dataset
            self.valid_lists[dataset_index] = ls
            # update corresponding info
            self.anno_num = self.anno_num - self.length_list[dataset_index] + anno_num
            self.length_list[dataset_index] = anno_num
            self.dataset_infos[dataset_index]['length'] = anno_num

    # 均匀点采样实验函数
    # 为每个语义点，从数据集中选出sample num个注释
    # 返回一个(anno_num,26)的列表，表示数据集中所有的可见性的值
    def __keypoint_sample(self, args):
        selected_vis_path = args.selected_vis_path
        all_vis = []
        print("visibility processing")
        # vis处理为每个semantic keypoint只有3000个点
        if not os.path.exists(selected_vis_path):
            func = OriginalLabelFormatTrans(extend_flag=True)
            print("random visibility generating")
            # all_vis -> (anno_num , 26)
            for valid_list in self.valid_lists:
                anns = valid_list['annotations']
                for ann in anns:
                    ann = func(ann)
                    ann_vis = [1 if tmp_ann_vis > 0 else 0 for tmp_ann_vis in ann['visible']]
                    all_vis.append(ann_vis)

            # counting for every semantic keypoint
            counts = [0] * 26
            # 统计每个部位可见性为1的数量
            for vis in all_vis:
                counts = [x + y for x, y in zip(vis, counts)]
            # 计算每个部位可见性为1的点需要减少的数量
            reduction_counts = [count - args.sample_num for count in counts]
            # 随机将vis置0而不是按照顺序
            sequence = list(range(self.anno_num))
            random.shuffle(sequence)
            for sequence_id in sequence:
                tmp_vis = all_vis[sequence_id]
                for tmp_ind, tmp_kp_vis in enumerate(tmp_vis):
                    if tmp_kp_vis > 0 and reduction_counts[tmp_ind] > 0:
                        tmp_vis[tmp_ind] = 0
                        reduction_counts[tmp_ind] -= 1
            res = {"info": "mix_sample_vis_value", "dataset_info": self.dataset_infos, "sample_num": args.sample_num,
                   "vis": all_vis}
            with open(selected_vis_path, 'w') as f:
                json.dump(res, f, indent=4)
        else:
            print("visibility loading")
            with open(selected_vis_path, 'r') as f:
                sample_vis_data = json.load(f)
            all_vis = sample_vis_data['vis']
        return all_vis

    # 根据keypoint_sample返回的所有注释的可见性的值更新数据集的valid_lists
    def __update_vis(self, args, all_vis):
        # all_vis now is anno_num * 26
        # change the vis value in self.valid_lists
        # 将新的vis值写回数据集的valid_lists
        reverse_func = NoLabelFormatTrans(extend_flag=False)
        for index, single_vis in enumerate(all_vis):
            inner_index = copy.deepcopy(index)
            for i, length in enumerate(self.length_list):
                # 判断inner_index对应的anno属于哪个dataset
                if inner_index <= length - 1:
                    # 将(26,)转换回原本的格式
                    current_dataset = self.dataset_infos[i]['dataset']
                    current_vis = reverse_func(single_vis, dataset=current_dataset)
                    current_vis = [int(x) for x in current_vis]
                    self.valid_lists[i]['annotations'][inner_index]['visible'] = np.array(current_vis)
                    break
                else:
                    inner_index -= length
        print("visibility wrote back done")

    # 根据keypoint_sample返回的所有注释的可见性的值，消除数据集中可见性值全为0的注释
    def __valid_ids_get(self, args, all_vis):
        print("generating valid ids")
        res_info = []
        if not os.path.exists(args.selected_info_path):
            # throw the labels without valid keypoints
            # valid ids for every dataset
            valid_ids = [[] for i in range(len(self.dataset_infos))]
            # 将all_vis分成几个部分
            all_split_vis = []
            start_index = 0
            end_index = 0
            for dataset_index, dataset_length in enumerate(self.length_list):
                end_index = start_index + dataset_length
                tmp_joints_vis = all_vis[start_index:end_index]
                all_split_vis.append(tmp_joints_vis)
                start_index = end_index

            for dataset_index, dataset_vis_info in enumerate(all_split_vis):
                for info_index, vis_info in enumerate(dataset_vis_info):
                    if np.sum(vis_info) > 0:
                        anno_id = self.valid_lists[dataset_index]['annotations'][info_index]['anno_id']
                        valid_ids[dataset_index].append(anno_id)
                tmp_info_dict = {"dataset": self.valid_lists[dataset_index]['dataset'],
                                 "mode": self.valid_lists[dataset_index]['mode'],
                                 "anno_ids": valid_ids[dataset_index], "anno_num": len(valid_ids[dataset_index])}
                res_info.append(tmp_info_dict)

            with open(args.selected_info_path, 'w') as f:
                json.dump(res_info, f, indent=4)
        else:
            with open(args.selected_info_path, 'r') as f:
                res_info = json.load(f)
        print("valid ids done")
        return res_info

    def __filter_by_valid_ids(self, res_info):
        print("start eliminating blank labels")
        for dataset_index, ls in enumerate(self.valid_lists):
            current_dataset = ls['dataset']
            current_mode = ls['mode']
            current_anns = ls['annotations'].copy()
            current_id_set = set()
            # generate its corresponding id set
            for info in res_info:
                if info['dataset'] == current_dataset and info['mode'] == current_mode:
                    ls['length'] = info['anno_num']
                    for one_id in info['anno_ids']:
                        if one_id not in current_id_set:
                            current_id_set.add(one_id)
                    break
            # 过滤注释列表
            current_anns = [ann for ann in current_anns if ann['anno_id'] in current_id_set]
            ls['annotations'] = current_anns

            anno_num = ls['length']
            # update the list to the original dataset
            self.valid_lists[dataset_index] = ls
            # update corresponding info
            self.anno_num = self.anno_num - self.length_list[dataset_index] + anno_num
            self.length_list[dataset_index] = anno_num
            self.dataset_infos[dataset_index]['length'] = anno_num
        print("elimination of blank labels has been done")

    def __filter_by_invalid_ids(self, res_info):
        print("start eliminating blank labels")
        for dataset_index, ls in enumerate(self.valid_lists):
            current_dataset = ls['dataset']
            current_mode = ls['mode']
            current_anns = ls['annotations'].copy()
            current_id_set = set()
            # generate its corresponding id set
            for info in res_info:
                if info['dataset'] == current_dataset and info['mode'] == current_mode:
                    # for invalid
                    ls['length'] -= info['anno_num']
                    for one_id in info['anno_ids']:
                        if one_id not in current_id_set:
                            current_id_set.add(one_id)
                    break
            # 过滤注释列表
            current_anns = [ann for ann in current_anns if ann['anno_id'] not in current_id_set]
            ls['annotations'] = current_anns

            anno_num = ls['length']
            # update the list to the original dataset
            self.valid_lists[dataset_index] = ls
            # update corresponding info
            self.anno_num = self.anno_num - self.length_list[dataset_index] + anno_num
            self.length_list[dataset_index] = anno_num
            self.dataset_infos[dataset_index]['length'] = anno_num
        print("elimination of blank labels has been done")

    def uniform_sample(self, args):
        all_vis = self.__keypoint_sample(args)
        self.__update_vis(args, all_vis)
        res_info = self.__valid_ids_get(args, all_vis)
        self.__filter_by_valid_ids(res_info)

    def blank_sample(self, args):
        all_vis = self.__keypoint_sample(args)
        self.__update_vis(args, all_vis)
        res_info = self.__valid_ids_get(args, all_vis)
        self.__filter_by_invalid_ids(res_info)

    def sample_few_by_dataset(self, args, num_lamda=0.1):
        for list_index, valid_list in enumerate(self.valid_lists):
            valid_anns = valid_list['annotations']
            num = int(len(valid_anns) * num_lamda)
            val_list = random.sample(valid_anns, num)
            valid_list['annotations'] = val_list
            # 更新相关的信息
            self.valid_lists[list_index]['length'] = num
            self.anno_num = self.anno_num - self.length_list[list_index] + num
            self.length_list[list_index] = num
            self.dataset_infos[list_index]['length'] = num
        self.get_kps_num(args, num_lamda)

    def sample_few(self,num_ratio=0.1):
        select_num = int(self.anno_num * num_ratio)  # 计算要生成的随机索引数量
        random_indices = random.sample(range(self.anno_num), select_num)  # 从0到anno_num-1的范围生成指定数量的随机索引
        # 将整体的index转换成局部数据集的index
        local_indices = [[] for _ in range(len(self.valid_lists))]
        for index in random_indices:
            tmp_index = index
            for i, length in enumerate(self.length_list):
                if tmp_index <= length - 1:
                    local_indices[i].append(tmp_index)
                    break
                else:
                    tmp_index -= length
        # update self.valid_lists by local_indices
        for dataset_index in range(len(self.valid_lists)):
            tmp_anns = self.valid_lists[dataset_index]["annotations"]
            tmp_indices = local_indices[dataset_index]
            tmp_anns = [ann for j,ann in enumerate(tmp_anns) if j in tmp_indices]
            self.valid_lists[dataset_index]["annotations"] = tmp_anns
        self.__update_info()
        # self.get_kps_num(args,num_ratio)

    def balance_tigdog(self, args, num_lamda=0.1):
        for list_index, valid_list in enumerate(self.valid_lists):
            if valid_list['dataset'] == 'tigdog_horse' or valid_list['dataset'] == 'tigdog_tiger':
                valid_anns = valid_list['annotations']
                num = int(len(valid_anns) * num_lamda)
                val_list = random.sample(valid_anns, num)
                valid_list['annotations'] = val_list
                # 更新相关的信息
                self.valid_lists[list_index]['length'] = num
                self.anno_num = self.anno_num - self.length_list[list_index] + num
                self.length_list[list_index] = num
                self.dataset_infos[list_index]['length'] = num
        self.get_kps_num(args, num_lamda)

    def get_kps_num(self, args):
        # 统计每个点对应的数量，并保存其信息到本地的statistics文件中
        with open(args.keypoints_path) as f:
            kps = json.load(f)['keypoints']
        kp_num = [0 for _ in range(self.num_joints)]
        res = {'keypoints': {}}
        if args.num_joints == 26:
            trans = OriginalLabelFormatTrans(extend_flag=True)
        else:
            trans = OriginalLabelFormatTransAP10KAnimalPose(extend_flag=True)

        for list_index, valid_list in enumerate(self.valid_lists):
            tmp_num = [0 for _ in range(self.num_joints)]
            labels = valid_list['annotations']
            current_dataset = valid_list['dataset']
            current_mode = valid_list['mode']
            for label in labels:
                # 应先转换到统一的26个点的模板再进行统计
                tmp_label = copy.deepcopy(label)
                tmp_label = trans(tmp_label)
                vis = tmp_label['visible']
                for val_index, val in enumerate(vis):
                    if val > 0:
                        kp_num[val_index] += 1
                        tmp_num[val_index] += 1
            res[current_dataset] = {current_mode: {}}
            res[current_dataset][current_mode] = {key: val for key, val in zip(kps, tmp_num)}

        res['keypoints'] = {key: val for key, val in zip(kps, kp_num)}
        # print("sample kps num :", res)
        logger.info("sample kps num :%s", res)

        # write_path = f"../statistics_file/num_results/ap10k_animalpose_train_dataset_kp_num.json"
        # with open(write_path, 'w') as f:
        #     json.dump(res, f, indent=4)

    # 使merge后某一点的数量保持不变。 5 + 7 = 12 保持不变则从merge后的12个里面随机选5
    # 但如果是 10 + 90 = 100 随机选10...
    # 可以对不同混合比例进行对比 完全随机 / 2 : 1 / 1 : 1 / 1 : 2
    def __mix_kps_num_get(self, args):
        num_mix = [0] * 26
        # map to 26
        with open(args.keypoints_path, 'r') as f:
            mix_definition = json.load(f)
        # 统计合并后每个点的数量 num_mix
        # 对于关注的点，随机抹去其有效点的可见性
        for single_dataset_index, single_dataset in enumerate(self.dataset_infos):
            current_dataset = single_dataset['dataset']
            current_mode = single_dataset['mode']
            current_num_joints = single_dataset['num_joints']
            current_num = [0] * current_num_joints
            map_info = mix_definition[current_dataset]['map']
            current_coco = self.coco_lists[single_dataset_index]['coco']
            for img_id in current_coco.getImgIds():
                img = current_coco.loadImgs(img_id)[0]
                ann_ids_gt = current_coco.getAnnIds(imgIds=img['id'])
                anns_gt = current_coco.loadAnns(ann_ids_gt)
                for i in range(len(anns_gt)):
                    ann_gt = anns_gt[i]
                    # list ( 51, )
                    kps = ann_gt['keypoints']
                    vis = kps[2::3]
                    num_tmp = [1 if vi > 0 else 0 for vi in vis]
                    current_num = [a + b for a, b in zip(current_num, num_tmp)]
            # 将当前数据集的点的数量映射到mix的格式上
            for pair in map_info:
                k, v = pair
                num_mix[v] += current_num[k]
            print("===========kps num info=========")
            print(current_dataset, current_mode)
            print(current_num)
        return num_mix

    # 随机抹掉指定数量制定点的可见性值为0
    # 有23个注释在经过目标关键点的可见性抹去之后变成了无效注释
    def __invalid_ids_get(self, args, kps_num, index_list, num_list):
        with open(args.keypoints_path, 'r') as f:
            mix_definition = json.load(f)

        eliminate_num = []
        for a, b in zip(index_list, num_list):
            ori_num = kps_num[a]
            eliminate_num.append(ori_num - b)
        ids = [i for i in range(self.anno_num)]
        random.shuffle(ids)

        invalid_dict = {}
        for info in self.dataset_infos:
            invalid_dict[info['dataset']] = {info['mode']: []}

        # 找到对应的数据 如果指定点可见性为1，将其设为0
        # 有的数据可能变成无效ann,注意更新数据集的信息
        for single_id in ids:
            for i, length in enumerate(self.length_list):
                if single_id <= length - 1:
                    target = self.valid_lists[i]['annotations'][single_id]
                    dataset = self.valid_lists[i]['dataset']
                    mode = self.valid_lists[i]['mode']
                    map_info = mix_definition[dataset]['map']
                    # 将26点定义为基础的index_list转换为对应的index (17,19,20)
                    swapped_map_info = [[y, x] for x, y in map_info]
                    tmp_index = []
                    for mix_index in index_list:
                        for pair in map_info:
                            if pair[-1] == mix_index:
                                tmp_index.append(pair[0])
                                break

                    for j, batch in enumerate(zip(tmp_index, eliminate_num)):
                        index, num = batch
                        if target['visible'][index] > 0 and num > 0:
                            target['visible'][index] = 0
                            eliminate_num[j] -= 1
                    if np.sum(target['visible']) == 0:
                        invalid_dict[dataset][mode].append(single_id)
                    break
                else:
                    single_id -= length

        return invalid_dict

    # update info by self.valid_lists
    def __update_info(self):
        num = []
        for valid_list in self.valid_lists:
            num.append(len(valid_list['annotations']))
        self.length_list = num
        for i, single_num in enumerate(num):
            self.dataset_infos[i]['length'] = single_num
            self.valid_lists[i]['length'] = single_num
        self.anno_num = np.sum(num)

    # index_list : index for keypoints sampling
    # num_list : num for sampling
    def specific_kp_sample(self, args, index_list, num_list):
        kps_num = self.__mix_kps_num_get(args=args)
        invalid_ids_dict = self.__invalid_ids_get(args, kps_num, index_list, num_list)
        for list_index, valid_list in enumerate(self.valid_lists):
            current_dataset = valid_list['dataset']
            current_mode = valid_list['mode']
            current_ids = sorted(invalid_ids_dict[current_dataset][current_mode], reverse=False)
            # filtered_list = [my_list[i] for i in valid_indices]
            anns = self.valid_lists[list_index]['annotations']
            anns = [ann for i, ann in enumerate(anns) if i not in current_ids]
            self.valid_lists[list_index]['annotations'] = anns
        self.__update_info()

    # unlabel dataset根据label dataset去除部分重叠数据
    def eliminate_repeated_data(self,label_valid_lists):
        # 收集label dataset中数据的id
        idxs = [[] for _ in range(len(label_valid_lists))]
        for list_index,valid_list in enumerate(label_valid_lists):
            anns = valid_list['annotations']
            for ann in anns:
                idxs[list_index].append(ann['anno_id'])
        # 根据 label dataset的标签id消除Unlabel dataset中重复的注释
        for ids,valid_list in zip(idxs,self.valid_lists):
            anns = valid_list['annotations']
            anns = [ann for ann in anns if ann['anno_id'] not in ids]
            valid_list['annotations'] = anns
        self.__update_info()

    def sample_animal_few_shot(self,sample_num=None):
        assert len(sample_num) == len(self.valid_lists), "Length of Sample Num should be same as the number of dataset."
        for i,valid_list in enumerate(self.valid_lists):
            cats = [0 for _ in range(len(self.coco_lists[i]['coco'].cats)+1)]
            cur_ann_ids = []
            cur_sample_num = sample_num[i]
            anns = valid_list['annotations']
            anns = sorted(anns,key=lambda x:np.count_nonzero(x['visible']),reverse=True)
            for ann in anns:
                category_id = ann['category_id']
                if cats[category_id] < cur_sample_num:
                    cur_ann_ids.append(ann['anno_id'])
                    cats[category_id] += 1
            anns = [ann for ann in anns if ann['anno_id'] in cur_ann_ids]
            anns = sorted(anns,key=lambda x:x['category_id'],reverse=False)
            valid_list['annotations'] = anns
        self.__update_info()

    def eliminate_specific_animals(self,dataset_index,animal_indices,save_num):
        anns = self.valid_lists[dataset_index]['annotations']
        # get the animal num
        cat_num = [0 for _ in range(len(self.coco_lists[dataset_index]['coco'].cats)+1)]
        for ann in anns:
            cat_num[ann['category_id']] += 1
        # eliminate the target animal anns
        to_del_indices = []
        for i,ann in enumerate(anns):
            if ann['category_id'] in animal_indices and cat_num[ann['category_id']] > save_num:
                to_del_indices.append(i)
                cat_num[ann['category_id']] = max(0,cat_num[ann['category_id']]-1)
        to_del_indices = sorted(to_del_indices,key=lambda x:x,reverse=True)
        for ind in to_del_indices:
            del anns[ind]
        self.__update_info()

    # save current image id into local file
    # image id bias for Animal Pose Dataset is 58632
    def save_img_ids(self):
        img_ann_ids = {}
        img_ids = []
        for valid_list in self.valid_lists:
            anns = valid_list['annotations']
            cur_dataset = valid_list['dataset']
            if cur_dataset == 'ap_10k':
                img_id_bias = 0
                ann_id_bias = 0
            else:
                img_id_bias = 58632
                ann_id_bias = 16561
            for ann in anns:
                cur_img_id = int(ann['image_id'] + img_id_bias)
                cur_ann_id = int(ann['anno_id'] + ann_id_bias)
                if cur_img_id not in img_ann_ids:
                    img_ann_ids[cur_img_id] = []
                img_ann_ids[cur_img_id].append(cur_ann_id)
                if cur_img_id not in img_ids:
                    img_ids.append(cur_img_id)
        imgs_num = len(img_ids)
        img_ann_ids_save_path = f'./info/random_{imgs_num}_img_ann_ids.json'
        img_ids_save_path = f'./info/random_{imgs_num}_img_ids.json'
        with open(img_ann_ids_save_path,'w') as f:
            json.dump(img_ann_ids,f,ensure_ascii=False)
        with open(img_ids_save_path,'w') as f:
            json.dump(img_ids,f)
        print(f'Files wrote done: {img_ids_save_path}')
        print(f'Files wrote done: {img_ann_ids_save_path}')

    def tmp_method(self):
        ap_10k_path = '../info/4_anns_ap_10k_ids.json'
        animal_pose_path = '../info/4_anns_animal_pose_ids.json'
        with open(ap_10k_path, 'r') as f:
            ap_10k_ids = json.load(f)
        with open(animal_pose_path, 'r') as f:
            animal_pose_ids = json.load(f)
        anno_ids_ap_10k = [info['anno_id'] for info in ap_10k_ids]
        anno_ids_animal_pose = [info['anno_id'] for info in animal_pose_ids]
        ap_10k_anns = self.valid_lists[0]['annotations']
        animal_pose_anns = self.valid_lists[1]['annotations']
        self.valid_lists[0]['annotations'] = [ann for ann in ap_10k_anns if ann['anno_id'] in anno_ids_ap_10k]
        self.valid_lists[1]['annotations'] = [ann for ann in animal_pose_anns if ann['anno_id'] in anno_ids_animal_pose]
        self.__update_info()

    def load_anns_from_file(self,path):
        with open(path,'r') as f:
            ids_info = json.load(f)
        ap_10k_img_id_bias = 58632
        ap_10k_ann_id_bias = 16561
        ap_10k_ann_id_set = set()
        animal_pose_ann_id_set = set()
        for img_id in ids_info:
            tmp_ann_ids = np.copy(ids_info[img_id])
            # if it is animal pose anns
            if int(img_id) > ap_10k_img_id_bias:
                for i,ann_id in enumerate(tmp_ann_ids):
                    tmp_ann_ids[i] = ann_id - ap_10k_ann_id_bias
                for val in tmp_ann_ids:
                    animal_pose_ann_id_set.add(val)
            else:
                for val in tmp_ann_ids:
                    ap_10k_ann_id_set.add(val)
        for valid_list in self.valid_lists:
            if valid_list['dataset'] == 'ap_10k':
                anno_ids_set = ap_10k_ann_id_set
            elif valid_list['dataset'] == 'animal_pose':
                anno_ids_set = animal_pose_ann_id_set
            else:
                logger.error("No matching dataset")
                break
            annotations = valid_list['annotations']
            valid_list['annotations'] = [ann for ann in annotations if ann['anno_id'] in anno_ids_set]
        self.__update_info()

    @staticmethod
    def collate_fn(batch):
        imgs_tuple, targets_tuple = tuple(zip(*batch))
        imgs_tensor = torch.stack(imgs_tuple)
        return imgs_tensor, targets_tuple

    @staticmethod
    def collate_fn_mpl(batch):
        imgs_list, targets_list = list(zip(*batch))
        ori_imgs = [imgs[0] for imgs in imgs_list]
        aug_imgs = [imgs[1] for imgs in imgs_list]
        ori_imgs_tensor = torch.stack(ori_imgs)
        aug_imgs_tensor = torch.stack(aug_imgs)
        return (ori_imgs_tensor,aug_imgs_tensor), targets_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--keypoints-path', default="./info/ap_10k_animal_pose_union_keypoints_format.json", type=str,
                        help='keypoints_format.json path')
    parser.add_argument('--fixed-size', default=[256, 256], nargs='+', type=int, help='input size')
    parser.add_argument('--num-joints', default=21, type=int, help='num_joints')

    parser.add_argument('--seed', default=2, type=int, help='seed for initializing training')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.keypoints_path, "r") as f:
        animal_kps_info = json.load(f)
    fixed_size = args.fixed_size
    heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)
    kps_weights = np.array(animal_kps_info["kps_weights"],
                           dtype=np.float32).reshape((args.num_joints,))

    data_transform = {
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
    train_label_dataset = MixKeypoint(root='../../dataset', merge_info=train_dataset_info, transform=data_transform['train'])

    train_label_dataset.sample_animal_few_shot(sample_num=[4,3])
    shared_animal_indices = [5,6,8,21,24]
    train_label_dataset.eliminate_specific_animals(dataset_index=0,animal_indices=shared_animal_indices,save_num=1)
    train_label_dataset.get_kps_num(args)
    train_label_dataset.save_img_ids()
    print('debug')
    # train_label_dataset.sample_few(num_ratio=0.02)
    # train_label_dataset.get_kps_num(args)
    # train_label_dataset.save_img_ids()
    # index_list = [0,1]
    # num_list = [1000,1000]
    # train_dataset.get_kps_num(args,1)
    # train_dataset.specific_kp_sample(args,index_list,num_list)
    # train_dataset.get_kps_num(args,2)

    # data_root = "../dataset"
    # dataset = 'ap_10k'
    # label_data_root = os.path.join(data_root, dataset)
    # train_dataset = CocoKeypoint(root=label_data_root, dataset=dataset, mode="train",transform=None, fixed_size=None,
    #                              data_type="keypoints")
    # # train_dataset.sample_anns_by_imgs_ids_file(path='./info/label_list/annotation_list_20.json')
    # train_dataset.get_qualified_balanced_few_shot(num_per_species=20)
    # print('debug')

    # train_dataset.get_avg_animals(20)
    # train_dataset.animal_category_num()
    # print('debug')
    # train_dataset_a = CocoKeypoint(root=label_data_root, dataset=dataset, mode="train",
    #                                transform=None, fixed_size=None, data_type="keypoints")
    # train_dataset_b = CocoKeypoint(root=label_data_root, dataset=dataset, mode="train",
    #                                transform=None, fixed_size=None, data_type="keypoints")
    # train_dataset_a.animal_category_num()
    # train_dataset_a.get_qualified_few_shot(num=912)
    # train_dataset_a.animal_category_num()
    # train_dataset_b.sample_few(args)
    # train_dataset_b.animal_category_num()
    # train_dataset.animal_avg_keypoint_num()
    # train_dataset.animal_gt_show(animal_category=None)
    # kp_index_list = [0,1]
    # train_dataset.block_specific_kp(args,kp_index_list)
    # label_data_root = "../dataset/tigdog"
    # train_dataset = CocoKeypoint(root=label_data_root, dataset="tigdog", mode="train",
    #                              transform=None,fixed_size=None, data_type="keypoints")
    # train_dataset.animal_gt_show(animal_category='horse')
