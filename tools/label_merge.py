import json
import pycocotools
import os
import numpy as np
import json
import random
import copy


def merge_labels():
    """
        合并AP-10K 和 Animal Pose的标签文件
        需要注意的是二者的anns和imgs的id分配重叠问题
        另外需要将Animal Pose的category字段向AP-10K对齐
        anno_id , id , img_id
        AP-10K Train + Animal Pose Train -> Merged Train
        AP-10K Val + Animal Pose Val -> Merged Val
        合并时不能单单考虑单独数据集中的id
        应该考虑train val test 所有数据集中的id情况
        将点转换为26或21个点对应的情况
        修改可见性的值 0 / 1
        将AP-10K train和Animal Pose train合并
        将AP-10K val和Animal Pose val合并
    """
    base_dir = '../info'
    datasets = ['ap_10k', 'animal_pose']
    ap_10k_train_path = os.path.join(base_dir, f"{datasets[0]}_train.json")
    ap_10k_val_path = os.path.join(base_dir, f"{datasets[0]}_val.json")
    ap_10k_test_path = os.path.join(base_dir, f"{datasets[0]}_test.json")
    animal_pose_train_path = os.path.join(base_dir, f"{datasets[1]}_train.json")
    animal_pose_val_path = os.path.join(base_dir, f"{datasets[1]}_val.json")

    ap_10k_paths = [ap_10k_train_path,ap_10k_val_path,ap_10k_test_path]

    ap_10k_imgs = []
    ap_10k_anns = []
    ap_10k_infos = []

    animal_pose_train_imgs = []
    animal_pose_val_imgs = []
    animal_pose_train_anns = []
    animal_pose_val_anns = []

    categoryAP10K2AnimalPose = [[8,1],[24,2],[6,3],[21,4],[5,5]]

    for i,path in enumerate(ap_10k_paths):
        with open(path,'r') as f:
            tmp_label_info = json.load(f)
        tmp_imgs = tmp_label_info['images']
        tmp_anns = tmp_label_info['annotations']
        ap_10k_imgs.extend(tmp_imgs)
        ap_10k_anns.extend(tmp_anns)
        ap_10k_infos.append(tmp_label_info)

    with open(animal_pose_train_path,'r') as f:
        animal_pose_train_label_info = json.load(f)
    tmp_imgs = animal_pose_train_label_info['images']
    tmp_anns = animal_pose_train_label_info['annotations']
    animal_pose_train_imgs.extend(tmp_imgs)
    animal_pose_train_anns.extend(tmp_anns)

    with open(animal_pose_val_path,'r') as f:
        animal_pose_val_label_info = json.load(f)
    tmp_imgs = animal_pose_val_label_info['images']
    tmp_anns = animal_pose_val_label_info['annotations']
    animal_pose_val_imgs.extend(tmp_imgs)
    animal_pose_val_anns.extend(tmp_anns)

    ap_10k_img_id_bias = max(ap_10k_imgs,key=lambda x:x['id'])['id']  # 58632
    ap_10k_ann_id_bias = max(ap_10k_anns,key=lambda x:x['id'])['id']  # 16561

    for img in animal_pose_train_imgs:
        img['id'] += ap_10k_img_id_bias
        img['from'] = 'animal_pose'
    for img in animal_pose_val_imgs:
        img['id'] += ap_10k_img_id_bias
        img['from'] = 'animal_pose'

    for ann in animal_pose_train_anns:
        ann['id'] += ap_10k_ann_id_bias
        ann['image_id'] += ap_10k_img_id_bias
        origin_category_id = ann['category_id']
        for category_pair in categoryAP10K2AnimalPose:
            if category_pair[1] == origin_category_id:
                ann['category_id'] = category_pair[0]
                break

    for ann in animal_pose_val_anns:
        ann['id'] += ap_10k_ann_id_bias
        ann['image_id'] += ap_10k_img_id_bias
        origin_category_id = ann['category_id']
        for category_pair in categoryAP10K2AnimalPose:
            if category_pair[1] == origin_category_id:
                ann['category_id'] = category_pair[0]
                break

    # 17 kps -> 21 kps   /   20 kps -> 21 kps
    convert_kps(animal_pose_val_anns,dataset='animal_pose')
    convert_kps(animal_pose_train_anns,dataset='animal_pose')

    ap_10k_train_anns = ap_10k_infos[0]['annotations']
    ap_10k_val_anns = ap_10k_infos[1]['annotations']
    convert_kps(ap_10k_train_anns,dataset='ap_10k')
    convert_kps(ap_10k_val_anns,dataset='ap_10k')

    merged_train_info = copy.deepcopy(ap_10k_infos[0])
    merged_val_info = copy.deepcopy(ap_10k_infos[1])
    merged_train_info['annotations'] = ap_10k_train_anns + animal_pose_train_anns
    for img in merged_train_info['images']:
        img['from'] = 'ap_10k'
    merged_train_info['images'] += animal_pose_train_imgs
    for img in merged_train_info['images']:
        del img['license']
    random.shuffle(merged_train_info['annotations'])
    random.shuffle(merged_train_info['images'])
    merged_train_info['info'] = {'description':'A Merged Training Dataset of AP-10K + Animal Pose Dataset',
                                 'year':2023,
                                 'contributor':'Yu ZHU',
                                 'date_created':'2023/10/17'}

    merged_val_info['annotations'] = ap_10k_val_anns + animal_pose_val_anns
    for img in merged_val_info['images']:
        img['from'] = 'ap_10k'
    merged_val_info['images'] += animal_pose_val_imgs
    for img in merged_val_info['images']:
        del img['license']
    random.shuffle(merged_val_info['annotations'])
    random.shuffle(merged_val_info['images'])
    merged_val_info['info'] = {'description':'A Merged Validating Dataset of AP-10K + Animal Pose Dataset',
                               'year':2023,
                               'contributor':'Yu ZHU',
                               'date_created':'2023/10/17'}

    train_info_path = '../info/ap_10k_animal_pose_union_train.json'
    val_info_path = '../info/ap_10k_animal_pose_union_val.json'
    with open(train_info_path,'w') as f:
        json.dump(merged_train_info,f)
    with open(val_info_path,'w') as f:
        json.dump(merged_val_info,f)

    print('write done')


def convert_kps(anns,dataset='ap_10k'):
    from info import ap_10k_animal_pose_union
    animal_pose_mapping = ap_10k_animal_pose_union.dataset_info['animal_pose_2_union']
    ap_10k_mapping = ap_10k_animal_pose_union.dataset_info['ap_10k_2_union']

    assert dataset in ['ap_10k','animal_pose']
    mapping = ap_10k_mapping if dataset == 'ap_10k' else animal_pose_mapping
    for ann in anns:
        ori_kps = ann['keypoints']
        tmp_kps = np.array(ori_kps).reshape(-1,3)
        for tmp_kp in tmp_kps:
            if tmp_kp[2] == 2:
                tmp_kp[2] = 1
        res_kps = np.zeros((21,3))
        for map_pair in mapping:
            res_kps[map_pair[1]] = tmp_kps[map_pair[0]]
        res_kps = list(res_kps.reshape(-1))
        res_kps = [int(val) for val in res_kps]
        ann['keypoints'] = res_kps
        if dataset == 'ap_10k':
            ann['from'] = 'ap_10k'
        elif dataset == 'animal_pose':
            ann['from'] = 'animal_pose'
        else:
            continue


if __name__ == '__main__':
    merge_labels()
