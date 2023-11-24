import os
from scipy import io
import numpy as np
import json
import cv2
from tqdm import tqdm

from train_utils.dataset import CocoKeypoint
from train_utils.transforms import scale_box


def inference():
    # base_dir_0 = "D:/AnimalPoseEstimation/dataset/ap_10k"
    # train_dataset_i = CocoKeypoint(base_dir_0, dataset="train", dataset_index=1)
    # anno_file_0 = os.path.join(base_dir_0,"annotations","ap10k-train.json")
    # with open(anno_file_0, 'r') as f:
    #     data_0 = json.load(f)

    base_dir = "D:/AnimalPoseEstimation/dataset/Animal-Pose Dataset/label_data"
    test_dataset = CocoKeypoint(base_dir)
    img_path = os.path.join(base_dir,"data")
    anno_file_1 = os.path.join(base_dir,"annotations","keypoints.json")
    # 读取 json 文件
    with open(anno_file_1, 'r') as f:
        data_1 = json.load(f)

    # tmp_dataset = CocoKeypoint_2(base_dir)

    # 获取 images 和 annotations 属性
    images = data_1['images']
    annotations = data_1['annotations']

    # 遍历 annotations 列表
    for annotation in annotations:
        # 获取 image_id 和 keypoints
        image_id = annotation['image_id']
        keypoints = annotation['keypoints']
        x,y,w,h = annotation['bbox']
        x2 = x+w
        y2 = y+h

        # 获取图像路径并读取图像
        image_path = images[str(image_id)]
        image = cv2.imread(os.path.join(img_path,image_path))
        print(image_path)
        height,width,_ = image.shape
        cv2.rectangle(image, (int(x), int(y)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
        # 遍历 keypoints 列表
        for i in range(0, len(keypoints)):
            # 获取关键点坐标和可见性
            x, y, v = keypoints[i]

            # 如果关键点可见，则在图像上绘制关键点和文本标签
            if v > 0:
                # 绘制关键点
                cv2.circle(image, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)

                # 绘制文本标签
                cv2.putText(image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)
        # print(keypoints
        print(f"wdith:{width},x:{x},x_2:{x2},height:{height},y:{y},y_2:{y2}")
        # # 创建支持调整大小的窗口
        # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        # # 显示图像
        # cv2.imshow('Image', image)
        cv2.imshow('Image',image)
        cv2.waitKey(0)


# for MPII Animal-Pose dataset
def json_format_convert():
    # 读取原始 JSON 文件
    with open('./keypoints.json', 'r') as f:
        data = json.load(f)

    # 创建新的 JSON 对象
    coco_data = {'info': {
        'description': 'animal pose dataset to COCO dataset format',
        'version': '1.0',
        'year': 2023,
        'contributor': 'zy',
        'date_created': '2023/07/17'
    }, 'licenses': []}

    base_img_dir = "../../dataset/animal_pose"
    img_dir = os.path.join(base_img_dir,"data")
    # 添加 images 部分
    coco_data['images'] = []
    for image_id, file_name in data['images'].items():
        # 读取图像
        image = cv2.imread(os.path.join(img_dir,file_name))
        # 获取图像尺寸
        height, width, _ = image.shape
        coco_data['images'].append({
            'id': int(image_id),
            'file_name': file_name,
            'width': width,  # Set the image width
            'height': height,  # Set the image height
            'license': 1
        })

    # 添加 annotations 部分
    coco_data['annotations'] = []
    for i, annotation in enumerate(data['annotations']):
        # src_keypoint = annotation['keypoints']
        # dst_keypoint = []
        # # mapping relation between ap-10k and Animal pose dataset
        # map_info = [[0,0],[1,1],[2,2],[17,3],[19,4],[5,5],[9,6],[13,7],[6,8],
        #             [10,9],[14,10],[7,11],[11,12],[15,13],[8,14],[12,15],[16,16],[3,17],[4,18],[18,19]]
        # num_points = 0
        # for rel in map_info:
        #     point = src_keypoint[rel[0]]
        #     dst_keypoint.append(point)
        #     if point[2] > 0:
        #         num_points +=1

        num_keypoints = 0
        kps = annotation['keypoints']
        for j,point in enumerate(kps):
            if point[0] != 0 or point[1] != 0:
                kps[j][-1] += 1
                num_keypoints += 1
        coco_data['annotations'].append({
            'id': i + 1,
            'image_id': annotation['image_id'],
            'category_id': annotation['category_id'],
            'bbox': annotation['bbox'],
            'area': annotation['bbox'][2]*annotation['bbox'][3],  # Set the area of the bounding box
            'iscrowd': 0,
            'keypoints': [coord for point in kps for coord in point],
            'num_keypoints': num_keypoints
        })

    # 添加 categories 部分
    coco_data['categories'] = data['categories']

    # 写入新的 JSON 文件
    with open('./animal_pose_coco.json', 'w') as f:
        json.dump(coco_data, f)


# for TigDog dataset
def mat_info_load(coco_list,animal_name):
    assert animal_name in ['horse','tiger']
    ranges = io.loadmat(f'../../dataset/TigDog/ranges/{animal_name}/ranges.mat')
    for info in ranges['ranges']:
        file_id = info[0]
        anns_num = info[2] - info[1] + 1
        file_path = f"../../dataset/TigDog/landmarks/{animal_name}/{file_id}.mat"
        if os.path.exists(file_path):
            file_anns = io.loadmat(file_path)
            for i in range(anns_num):
                ann = file_anns['landmarks'][i][0][0][0]
                img_name_id = f"{info[1] + i:08}"
                keypoints = ann[0]
                visible = ann[1]
                if animal_name == 'horse':
                    category_id = 1
                    img_name = f"horse/{img_name_id}.jpg"
                else:
                    category_id = 2
                    img_name = f"tiger/{img_name_id}.jpg"
                    # tiger 和 horse 点的定义不同
                    # 这里需要变化以下顺序
                    tmp_hip_kps = keypoints[8:12].copy()
                    keypoints[8:12] = keypoints[14:18]
                    keypoints[14:18] = tmp_hip_kps
                tmp_dict = {
                    'image_id':len(coco_list)+1,
                    'images_path':img_name,
                    'keypoints':keypoints,
                    'visible':visible,
                    'category_id':category_id
                }

                coco_list.append(tmp_dict)


def mat_format_convert():
    coco_list = []
    mat_info_load(coco_list,'horse')
    mat_info_load(coco_list,'tiger')
    # 创建新的 JSON 对象
    coco_data = {'info': {
        'description': 'TigDog dataset to COCO dataset format',
        'version': '1.0',
        'year': 2023,
        'contributor': 'zy',
        'date_created': '2023/07/22'
    }, 'licenses': []}

    base_img_dir = "../../dataset/TigDog"

    # 添加 images 部分
    coco_data['images'] = []
    coco_data['annotations'] = []

    img_id_set = set()

    for ann in tqdm(coco_list):
        # 处理images字段
        img_path = ann['images_path']

        # 读取图像
        image = cv2.imread(os.path.join(base_img_dir, img_path))
        # 获取图像尺寸
        height, width, _ = image.shape

        if ann['image_id'] not in img_id_set:
            coco_data['images'].append({
                'id': int(ann['image_id']),
                'file_name': img_path,
                'width': width,  # Set the image width
                'height': height,  # Set the image height
                'license': 1
            })
            img_id_set.add(ann['image_id'])
        # 处理annotations字段
        visibility = ann['visible']
        num_keypoints = np.count_nonzero(visibility)
        kps = np.hstack((ann['keypoints'],ann['visible']))

        # xmin,ymin,xmax,ymax info for bbox
        valid_kps = ann['keypoints'][visibility.flatten().astype(bool)]
        if valid_kps.size > 0:
            x_min = int(np.amin(valid_kps[:,0]))
            y_min = int(np.amin(valid_kps[:,1]))
            x_max = int(np.amax(valid_kps[:,0]))
            y_max = int(np.amax(valid_kps[:,1]))

            w = x_max - x_min
            h = y_max - y_min
            if w > 1 and h > 1:
                # 把w和h适当放大点，要不然关键点处于边缘位置
                x_min, y_min, w, h = scale_box(x_min, y_min, w, h, (1.5, 1.5))
                x_min = int(max(0,x_min))
                y_min = int(max(0,y_min))
                w = int(min(width - x_min,w))
                h = int(min(height - y_min,h))
                x_max = x_min + w
                y_max = y_min + h

            # cv2.imshow("img",image)
            # # 在图像上绘制边界框
            # cv2.rectangle(image, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)
            #
            # # 显示图像
            # cv2.imshow("img", image)
            # cv2.waitKey(0)
        else:
            # 处理没有有效关键点的情况
            # 示例：直接设定初始边界框坐标
            x_min, y_min, w, h = 0, 0, width, height
            print(f"{ann['image_id']} no valid kps")
            continue

        coco_data['annotations'].append({
            'id':len(coco_data['annotations']) + 1,
            'image_id': int(ann['image_id']),
            'category_id': ann['category_id'],
            'bbox': [x_min,y_min,w,h],
            'area': w * h,  # Set the area of the bounding box
            'iscrowd': 0,
            'keypoints': [int(coord) for point in kps for coord in point],
            'num_keypoints': num_keypoints
        })

    # 添加 categories 部分
    coco_data['categories'] = [
        {"id": 1, "name": "horse"},
        {"id": 2, "name": "tiger"}
    ]

    # 写入新的 JSON 文件
    with open('./tigdog_coco_kps_hw.json', 'w') as f:
        json.dump(coco_data, f,indent=4)


def test_load():
    with open('./tigdog_coco_ori_hw.json','r') as f:
        ori_data = json.load(f)
    with open('./tigdog_coco_kps_hw.json','r') as f:
        kps_data = json.load(f)

    print('da')


def check_duplicate_files(folder1, folder2):
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    # 提取文件名并创建集合
    filenames1 = set([f.split('.')[0] for f in files1])
    filenames2 = set([f.split('.')[0] for f in files2])

    # 检查共同的文件名
    duplicate_filenames = filenames1.intersection(filenames2)
    if len(duplicate_filenames) > 0:
        print("存在以下命名重复的文件：")
        for filename in duplicate_filenames:
            print(f"{len(duplicate_filenames)} {filename}")
    else:
        print("没有发现命名重复的文件。")


def json_convert_tigdog(path):
    with open(path,'r') as f:
        data = json.load(f)
    # 创建新的 JSON 对象
    coco_data = {'info': {
        'description': 'tigdog dataset to COCO dataset format',
        'version': '1.0',
        'year': 2023,
        'contributor': 'zy',
        'date_created': '2023/07/26'
    }, 'licenses': []}

    base_img_dir = "../../dataset/tigdog"
    img_dir = os.path.join(base_img_dir,"data")
    # 添加 images 部分
    coco_data['images'] = []
    # 添加 annotations 部分
    coco_data['annotations'] = []

    img_idxs_set = set()

    for ann in tqdm(data):
        # 读取图像
        image = cv2.imread(os.path.join(img_dir,ann['img_name']))
        # 获取图像尺寸
        height, width, _ = image.shape
        coco_data['images'].append({
            'id': ann['image_id'],
            'file_name': ann['img_name'],
            'width': width,  # Set the image width
            'height': height,  # Set the image height
            'license': 1
        })

        # xmin,ymin,xmax,ymax info for bbox
        kps = np.array(ann['keypoints'])
        valid_kps = kps.reshape(19,3)[kps[2::3].astype(bool)]
        if valid_kps.size > 0:
            x_min = int(np.amin(valid_kps[:, 0]))
            y_min = int(np.amin(valid_kps[:, 1]))
            x_max = int(np.amax(valid_kps[:, 0]))
            y_max = int(np.amax(valid_kps[:, 1]))

            w = x_max - x_min
            h = y_max - y_min
            if w > 1 and h > 1:
                # 把w和h适当放大点，要不然关键点处于边缘位置
                x_min, y_min, w, h = scale_box(x_min, y_min, w, h, (1.5, 1.5))
                x_min = int(max(0, x_min))
                y_min = int(max(0, y_min))
                w = int(min(width - x_min, w))
                h = int(min(height - y_min, h))
                x_max = x_min + w
                y_max = y_min + h

            # cv2.imshow("img",image)
            # # 在图像上绘制边界框
            # cv2.rectangle(image, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)
            #
            # # 显示图像
            # cv2.imshow("img", image)
            # cv2.waitKey(0)
        else:
            # 处理没有有效关键点的情况
            # 示例：直接设定初始边界框坐标
            x_min, y_min, w, h = 0, 0, width, height
            print(f"{ann['image_id']} no valid kps")
            continue

        if ann['category_id'] == 2:
            # tiger 和 horse 点的定义不同
            # 这里需要变化以下顺序
            kps = kps.reshape(-1,3)
            tmp_hip_kps = kps[8:12].copy()
            kps[8:12] = kps[14:18]
            kps[14:18] = tmp_hip_kps
            kps = kps.reshape(-1)

        if ann['image_id'] in img_idxs_set:
            is_crowd = 1
        else:
            is_crowd = 0
            img_idxs_set.add(ann['image_id'])

        coco_data['annotations'].append({
            'id': ann['id'],
            'image_id': ann['image_id'],
            'category_id': ann['category_id'],
            'bbox': [x_min,y_min,w,h],
            'area': w * h,  # Set the area of the bounding box
            'keypoints': [int(val) for val in kps],
            'visible': [int(vis) for vis in kps[2::3]],
            'num_keypoints': ann['num_keypoints'],
            'iscrowd': is_crowd
        })

    # 添加 categories 部分
    coco_data['categories'] = [
        {"id": 1, "name": "horse"},
        {"id": 2, "name": "tiger"}
    ]

    # 写入新的 JSON 文件
    with open('./tigdog_val.json', 'w') as f:
        json.dump(coco_data, f,indent=4)


def tigdog_error_elimination():
    dataset = 'val'
    json_path = f'../../dataset/tigdog/annotations/tigdog_{dataset}.json'
    error_path = f'../show/errorLandmarks_{dataset}_error_type.json'
    output_path = f'../tigdog_{dataset}_correct.json'
    with open(json_path,'r') as f:
        label_data = json.load(f)
    with open(error_path,'r') as f:
        error_data = json.load(f)

    # 错误类型与注释索引的映射关系
    error_type_mapping = {
        1: [[12, 13]],
        2: [[16, 17]],
        3: [[12, 13],[16, 17]]
    }

    for error in error_data:
        error_id = error['id']
        error_type = error['error_type']

        # 获取错误类型对应的注释索引
        annotation_indices_list = error_type_mapping[error_type]

        for annotation in label_data['annotations']:
            if annotation['id'] == error_id:
                kps = np.array(annotation['keypoints']).reshape(19, 3)
                vis = np.array(annotation['visible']).copy()

                for index_list in annotation_indices_list:
                    # 交换关键点和可见性信息
                    temp_kps = kps[index_list[0]].copy()
                    temp_vis = vis[index_list[0]].copy()
                    kps[index_list[0]] = kps[index_list[1]]
                    vis[index_list[0]] = vis[index_list[1]]
                    kps[index_list[1]] = temp_kps
                    vis[index_list[1]] = temp_vis

                # 更新注释中的关键点和可见性信息
                annotation['keypoints'] = kps.flatten().tolist()
                annotation['visible'] = vis.tolist()

        # 保存修改后的数据集
    with open(output_path, 'w') as f:
        json.dump(label_data, f,indent=4)

    print('done')


def tigdog_split():
    mode = 'val'
    path = f'../../dataset/tigdog/annotations/tigdog_{mode}.json'
    out_horse_path = f'../tigdog_horse_{mode}.json'
    out_tiger_path = f'../tigdog_tiger_{mode}.json'
    with open(path,'r') as f:
        data = json.load(f)
    tiger_anno_ids = set()
    tiger_img_ids = set()
    horse_anno_ids = set()
    horse_img_ids = set()
    for anno in data['annotations']:
        if anno['category_id'] == 1:
            horse_anno_ids.add(anno['id'])
            horse_img_ids.add(anno['image_id'])
        else:
            tiger_anno_ids.add(anno['id'])
            tiger_img_ids.add(anno['image_id'])
    horse_split_data = {'info':data['info'],'licenses':[],'categories':data['categories'],'images':[],'annotations':[]}
    tiger_split_data = {'info':data['info'],'licenses':[],'categories':data['categories'],'images':[],'annotations':[]}
    for img in data['images']:
        if img['id'] in horse_img_ids:
            horse_split_data['images'].append(img)
        elif img['id'] in tiger_img_ids:
            img['id'] = len(tiger_split_data['images']) + 1
            tiger_split_data['images'].append(img)
        else:
            continue
    for ann in data['annotations']:
        if ann['id'] in horse_anno_ids:
            horse_split_data['annotations'].append(ann)
        elif ann['id'] in tiger_img_ids:
            ann['id'] = len(tiger_split_data['annotations']) + 1
            ann['image_id'] = len(tiger_split_data['annotations']) + 1
            tiger_split_data['annotations'].append(ann)
        else:
            continue
    print('debug')
    with open(out_horse_path,'w') as f:
        json.dump(horse_split_data,f,indent=4)
    with open(out_tiger_path,'w') as f:
        json.dump(tiger_split_data,f,indent=4)


if __name__ == '__main__':
    # json_format_convert()
    # inference()
    # mat_format_convert()
    # test_load()
    # 指定两个文件夹路径
    # folder1 = '../../dataset/TigDog/horse'
    # folder2 = "../../dataset/TigDog/tiger"
    #
    # # 调用函数检查重复文件
    # check_duplicate_files(folder1, folder2)
    # path_1 = 'tigdog_train_ori.json'
    # json_convert_tigdog(path_1)
    # path_2 = 'tigdog_val_ori.json'
    # json_convert_tigdog(path_2)
    # tigdog_error_elimination()
    tigdog_split()
