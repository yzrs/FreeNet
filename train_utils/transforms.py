import json
import math
import random
from typing import Tuple
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import copy

from train_utils.augmentation import RandWeakAugment,RandWeakAugmentFixMatch


def flip_images(img):
    assert len(img.shape) == 4, 'images has to be [batch_size, channels, height, width]'
    img = torch.flip(img, dims=[3])
    return img


def flip_back(output_flipped, matched_parts):
    assert len(output_flipped.shape) == 4, 'output_flipped has to be [batch_size, num_joints, height, width]'
    output_flipped = torch.flip(output_flipped, dims=[3])

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0]].clone()
        output_flipped[:, pair[0]] = output_flipped[:, pair[1]]
        output_flipped[:, pair[1]] = tmp

    return output_flipped


def flip_visible_back(input_flipped, matched_parts):
    assert len(input_flipped.shape) == 2, 'output_flipped has to be [batch_size, num_joints]'

    output_flipped = copy.deepcopy(input_flipped)
    for pair in matched_parts:
        tmp = output_flipped[:, pair[0]].clone()
        output_flipped[:, pair[0]] = output_flipped[:, pair[1]]
        output_flipped[:, pair[1]] = tmp

    return output_flipped


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'

    batch_size, num_joints, h, w = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    maxvals = maxvals.unsqueeze(dim=-1)
    idx = idx.float()

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps)

    preds[:, :, 0] = idx % w  # column 对应最大值的x坐标
    preds[:, :, 1] = torch.floor(idx / w)  # row 对应最大值的y坐标

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    preds *= pred_mask
    return preds, maxvals


def affine_points(pt, t):
    ones = np.ones((pt.shape[0], 1), dtype=float)
    pt = np.concatenate([pt, ones], axis=1).T
    new_pt = np.dot(t, pt)
    return new_pt.T


def get_final_preds(batch_heatmaps: torch.Tensor,
                    trans: list = None,
                    post_processing: bool = False):
    assert trans is not None
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if post_processing:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = torch.tensor(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    ).to(batch_heatmaps.device)
                    coords[n][p] += torch.sign(diff) * .25

    preds = coords.clone().cpu().numpy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = affine_points(preds[i], trans[i])

    return preds, maxvals.cpu().numpy()


def decode_keypoints(outputs, origin_hw, num_joints: int = 17):
    keypoints = []
    scores = []
    heatmap_h, heatmap_w = outputs.shape[-2:]
    for i in range(num_joints):
        pt = np.unravel_index(np.argmax(outputs[i]), (heatmap_h, heatmap_w))
        score = outputs[i, pt[0], pt[1]]
        keypoints.append(pt[::-1])  # hw -> wh(xy)
        scores.append(score)

    keypoints = np.array(keypoints, dtype=float)
    scores = np.array(scores, dtype=float)
    # convert to full image scale
    keypoints[:, 0] = np.clip(keypoints[:, 0] / heatmap_w * origin_hw[1],
                              a_min=0,
                              a_max=origin_hw[1])
    keypoints[:, 1] = np.clip(keypoints[:, 1] / heatmap_h * origin_hw[0],
                              a_min=0,
                              a_max=origin_hw[0])
    return keypoints, scores


def resize_pad(img: np.ndarray, size: tuple):
    h, w, c = img.shape
    src = np.array([[0, 0],  # 原坐标系中图像左上角点
                    [w - 1, 0],  # 原坐标系中图像右上角点
                    [0, h - 1]],  # 原坐标系中图像左下角点
                   dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    if h / w > size[0] / size[1]:
        # 需要在w方向padding
        wi = size[0] * (w / h)
        pad_w = (size[1] - wi) / 2
        dst[0, :] = [pad_w - 1, 0]  # 目标坐标系中图像左上角点
        dst[1, :] = [size[1] - pad_w - 1, 0]  # 目标坐标系中图像右上角点
        dst[2, :] = [pad_w - 1, size[0] - 1]  # 目标坐标系中图像左下角点
    else:
        # 需要在h方向padding
        hi = size[1] * (h / w)
        pad_h = (size[0] - hi) / 2
        dst[0, :] = [0, pad_h - 1]  # 目标坐标系中图像左上角点
        dst[1, :] = [size[1] - 1, pad_h - 1]  # 目标坐标系中图像右上角点
        dst[2, :] = [0, size[0] - pad_h - 1]  # 目标坐标系中图像左下角点

    trans = cv2.getAffineTransform(src, dst)  # 计算正向仿射变换矩阵
    # 对图像进行仿射变换
    resize_img = cv2.warpAffine(img,
                                trans,
                                size[::-1],  # w, h
                                flags=cv2.INTER_LINEAR)
    # import matplotlib.pyplot as plt
    # plt.imshow(resize_img)
    # plt.show()

    dst /= 4  # 网络预测的heatmap尺寸是输入图像的1/4
    reverse_trans = cv2.getAffineTransform(dst, src)  # 计算逆向仿射变换矩阵，方便后续还原

    return resize_img, reverse_trans


def adjust_box(xmin: float, ymin: float, w: float, h: float, fixed_size: Tuple[float, float]):
    """通过增加w或者h的方式保证输入图片的长宽比固定"""
    xmax = xmin + w
    ymax = ymin + h

    hw_ratio = fixed_size[0] / fixed_size[1]
    if h / w > hw_ratio:
        # 需要在w方向padding
        wi = h / hw_ratio
        pad_w = (wi - w) / 2
        xmin = xmin - pad_w
        xmax = xmax + pad_w
    else:
        # 需要在h方向padding
        hi = w * hw_ratio
        pad_h = (hi - h) / 2
        ymin = ymin - pad_h
        ymax = ymax + pad_h

    return xmin, ymin, xmax, ymax


def scale_box(xmin: float, ymin: float, w: float, h: float, scale_ratio: Tuple[float, float]):
    """根据传入的h、w缩放因子scale_ratio，重新计算xmin，ymin，w，h"""
    s_h = h * scale_ratio[0]
    s_w = w * scale_ratio[1]
    xmin = xmin - (s_w - w) / 2.
    ymin = ymin - (s_h - h) / 2.
    return xmin, ymin, s_w, s_h


def plot_heatmap(image, heatmap, kps, kps_weights):
    for kp_id in range(len(kps_weights)):
        if kps_weights[kp_id] > 0:
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.plot(*kps[kp_id].tolist(), "ro")
            plt.title("image")
            plt.subplot(1, 2, 2)
            plt.imshow(heatmap[kp_id], cmap=plt.cm.Blues)
            plt.colorbar(ticks=[0, 1])
            plt.title(f"kp_id: {kp_id}")
            plt.show()


class Compose(object):
    """组合多个transform函数"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = cv2.resize(image, self.size[::-1])
        # image = F.resize(image,self.size)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class HalfBody(object):
    def __init__(self, p: float = 0.3, upper_body_ids=None, lower_body_ids=None):
        assert upper_body_ids is not None
        assert lower_body_ids is not None
        self.p = p
        self.upper_body_ids = upper_body_ids
        self.lower_body_ids = lower_body_ids

    def __call__(self, image, target):
        if random.random() < self.p:
            kps = target["keypoints"]
            vis = target["visible"]
            upper_kps = []
            lower_kps = []

            # 对可见的keypoints进行归类
            for i, v in enumerate(vis):
                if v > 0.5:
                    if i in self.upper_body_ids:
                        upper_kps.append(kps[i])
                    else:
                        lower_kps.append(kps[i])

            # 50%的概率选择上或下半身
            if random.random() < 0.5:
                selected_kps = upper_kps
            else:
                selected_kps = lower_kps

            # 如果点数太少就不做任何处理
            if len(selected_kps) > 2:
                selected_kps = np.array(selected_kps, dtype=np.float32)
                xmin, ymin = np.min(selected_kps, axis=0).tolist()
                xmax, ymax = np.max(selected_kps, axis=0).tolist()
                w = xmax - xmin
                h = ymax - ymin
                if w > 1 and h > 1:
                    # 把w和h适当放大点，要不然关键点处于边缘位置
                    xmin, ymin, w, h = scale_box(xmin, ymin, w, h, (1.5, 1.5))
                    target["box"] = [xmin, ymin, w, h]

        return image, target


class AffineTransform(object):
    """scale+rotation"""

    def __init__(self,
                 scale: Tuple[float, float] = None,  # e.g. (0.65, 1.35)
                 rotation: Tuple[int, int] = None,  # e.g. (-45, 45)
                 fixed_size: Tuple[int, int] = (256, 256)):
        self.scale = scale
        self.rotation = rotation
        self.fixed_size = fixed_size

    def __call__(self, img, target):
        src_xmin, src_ymin, src_xmax, src_ymax = adjust_box(*target["box"], fixed_size=self.fixed_size)
        src_w = src_xmax - src_xmin
        src_h = src_ymax - src_ymin
        src_center = np.array([(src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2])
        src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
        src_p3 = src_center + np.array([src_w / 2, 0])  # right middle

        dst_center = np.array([(self.fixed_size[1] - 1) / 2, (self.fixed_size[0] - 1) / 2])
        dst_p2 = np.array([(self.fixed_size[1] - 1) / 2, 0])  # top middle
        dst_p3 = np.array([self.fixed_size[1] - 1, (self.fixed_size[0] - 1) / 2])  # right middle

        if self.scale is not None:
            scale = random.uniform(*self.scale)
            src_w = src_w * scale
            src_h = src_h * scale
            src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
            src_p3 = src_center + np.array([src_w / 2, 0])  # right middle

        if self.rotation is not None:
            angle = random.randint(*self.rotation)  # 角度制
            angle = angle / 180 * math.pi  # 弧度制
            src_p2 = src_center + np.array([src_h / 2 * math.sin(angle), -src_h / 2 * math.cos(angle)])
            src_p3 = src_center + np.array([src_w / 2 * math.cos(angle), src_w / 2 * math.sin(angle)])

        src = np.stack([src_center, src_p2, src_p3]).astype(np.float32)
        dst = np.stack([dst_center, dst_p2, dst_p3]).astype(np.float32)

        trans = cv2.getAffineTransform(src, dst)  # 计算正向仿射变换矩阵
        dst /= 4  # 网络预测的heatmap尺寸是输入图像的1/4
        reverse_trans = cv2.getAffineTransform(dst, src)  # 计算逆向仿射变换矩阵，方便后续还原

        # 对图像进行仿射变换
        resize_img = cv2.warpAffine(img,
                                    trans,
                                    tuple(self.fixed_size[::-1]),  # [w, h]
                                    flags=cv2.INTER_LINEAR
                                    )
        # from matplotlib import pyplot as plt
        # fig,axs = plt.subplots(1,2)
        # axs[0].imshow(img)
        # axs[1].imshow(resize_img)
        # plt.show()

        if "keypoints" in target:
            kps = target["keypoints"]
            mask = np.logical_and(kps[:, 0] != 0, kps[:, 1] != 0)
            kps[mask] = affine_points(kps[mask], trans)
            target["keypoints"] = kps

        target["trans"] = trans
        target["reverse_trans"] = reverse_trans
        return resize_img, target


class RandomHorizontalFlip(object):
    """随机对输入图片进行水平翻转，注意该方法必须接在 AffineTransform 后"""

    def __init__(self, p: float = 0.5, matched_parts: list = None):
        assert matched_parts is not None
        self.p = p
        self.matched_parts = matched_parts

    def __call__(self, image, target):
        if random.random() < self.p:
            # [h, w, c]
            image = np.ascontiguousarray(np.flip(image, axis=[1]))
            keypoints = target["keypoints"]
            visible = target["visible"]
            width = image.shape[1]

            # Flip horizontal
            keypoints[:, 0] = width - keypoints[:, 0] - 1

            # Change left-right parts
            for pair in self.matched_parts:
                keypoints[pair[0], :], keypoints[pair[1], :] = \
                    keypoints[pair[1], :], keypoints[pair[0], :].copy()

                visible[pair[0]], visible[pair[1]] = \
                    visible[pair[1]], visible[pair[0]].copy()

            target["keypoints"] = keypoints
            target["visible"] = visible

        return image, target


class KeypointToHeatMap(object):
    def __init__(self,
                 heatmap_hw: Tuple[int, int] = (192 // 4, 256 // 4),
                 gaussian_sigma: int = 2,
                 keypoints_weights=None):
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma
        self.kernel_radius = self.sigma * 3
        self.use_kps_weights = False if keypoints_weights is None else True
        self.kps_weights = keypoints_weights

        # generate gaussian kernel(not normalized)
        kernel_size = 2 * self.kernel_radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel[y, x] = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))
        # print(kernel)

        self.kernel = kernel

    def __call__(self, image, target):
        kps = target["keypoints"]
        num_kps = kps.shape[0]
        kps_weights = np.ones((num_kps,), dtype=np.float32)
        if "visible" in target:
            visible = target["visible"]
            kps_weights = visible

        heatmap = np.zeros((num_kps, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
        heatmap_kps = (kps / 4 + 0.5).astype(int)  # round
        for kp_id in range(num_kps):
            v = kps_weights[kp_id]
            if v < 0.5:
                # 如果该点的可见度很低，则直接忽略
                continue

            x, y = heatmap_kps[kp_id]
            ul = [x - self.kernel_radius, y - self.kernel_radius]  # up-left x,y
            br = [x + self.kernel_radius, y + self.kernel_radius]  # bottom-right x,y
            # 如果以xy为中心kernel_radius为半径的辐射范围内与heatmap没交集，则忽略该点(该规则并不严格)
            if ul[0] > self.heatmap_hw[1] - 1 or \
                    ul[1] > self.heatmap_hw[0] - 1 or \
                    br[0] < 0 or \
                    br[1] < 0:
                # If not, just return the image as is
                kps_weights[kp_id] = 0
                continue

            # Usable gaussian range
            # 计算高斯核有效区域（高斯核坐标系）
            g_x = (max(0, -ul[0]), min(br[0], self.heatmap_hw[1] - 1) - ul[0])
            g_y = (max(0, -ul[1]), min(br[1], self.heatmap_hw[0] - 1) - ul[1])
            # image range
            # 计算heatmap中的有效区域（heatmap坐标系）
            img_x = (max(0, ul[0]), min(br[0], self.heatmap_hw[1] - 1))
            img_y = (max(0, ul[1]), min(br[1], self.heatmap_hw[0] - 1))

            if kps_weights[kp_id] > 0.5:
                # 将高斯核有效区域复制到heatmap对应区域
                heatmap[kp_id][img_y[0]:img_y[1] + 1, img_x[0]:img_x[1] + 1] = \
                    self.kernel[g_y[0]:g_y[1] + 1, g_x[0]:g_x[1] + 1]

        if self.use_kps_weights:
            kps_weights = np.multiply(kps_weights, self.kps_weights)

        # plot_heatmap(image, heatmap, kps, kps_weights)

        target["heatmap"] = torch.as_tensor(heatmap, dtype=torch.float32)
        target["kps_weights"] = torch.as_tensor(kps_weights, dtype=torch.float32)

        return image, target


# transform dataset label format
# from ap_10k / animal_pose / tigdog ...  to mixed
# from mixed to ap_10k / animal_pose / tigdog
# src = 'ap_10k' des = 'mix'
class LabelFormatTrans(object):
    def __init__(self, extend_flag=True):
        self.extend_flag = extend_flag
        self.map_info = {'extend': {'ap_10k': {}, 'animal_pose': {},"tigdog":{}, "tigdog_horse":{},"tigdog_tiger":{}},
                         'shrink': {'ap_10k': {}, 'animal_pose': {},"tigdog":{}, "tigdog_horse":{},"tigdog_tiger":{}}
                         }
        # [ap_10k,mixed]
        ap_10k_map = [[0, 0], [1, 1], [2, 4], [3, 8], [4, 25], [5, 11], [6, 15], [7, 21],
                      [8, 12], [9, 16], [10, 22], [11, 13], [12, 17], [13, 23], [14, 14], [15, 18], [16, 24]]
        animal_pose_map = [[0, 0], [1, 1], [2, 4], [3, 2], [4, 3], [5, 11], [6, 12], [7, 13],
                           [8, 14], [9, 15], [10, 16], [11, 17], [12, 18], [13, 21], [14, 22], [15, 23],
                           [16, 24], [17, 6], [18, 7], [19, 25]]
        tigdog_map = [[0, 0], [1, 1], [2, 5], [3, 21], [4, 22], [5, 23], [6, 24], [7, 25],
                        [8, 15], [9, 16], [10, 17], [11, 18], [12, 9], [13, 10], [14, 11], [15, 12],
                        [16, 13], [17, 14], [18, 8]]
        tigdog_horse_map = [[0, 0], [1, 1], [2, 5], [3, 21], [4, 22], [5, 23], [6, 24], [7, 25],
                            [8, 15], [9, 16], [10, 17], [11, 18], [12, 9], [13, 10], [14, 11], [15, 12],
                            [16, 13], [17, 14], [18, 8]]
        tigdog_tiger_map = [[0, 0], [1, 1], [2, 5], [3, 21], [4, 22], [5, 23], [6, 24], [7, 25],
                            [8, 19], [9, 20], [10, 17], [11, 18], [12, 9], [13, 10], [14, 11], [15, 12],
                            [16, 13], [17, 14], [18, 8]]
        for lis in ap_10k_map:
            k, v = lis
            self.map_info['extend']['ap_10k'][k] = v
            self.map_info['shrink']['ap_10k'][v] = k
        for lis in animal_pose_map:
            k, v = lis
            self.map_info['extend']['animal_pose'][k] = v
            self.map_info['shrink']['animal_pose'][v] = k
        for lis in tigdog_horse_map:
            k, v = lis
            self.map_info['extend']['tigdog_horse'][k] = v
            self.map_info['shrink']['tigdog_horse'][v] = k
        for lis in tigdog_map:
            k, v = lis
            self.map_info['extend']['tigdog'][k] = v
            self.map_info['shrink']['tigdog'][v] = k
        for lis in tigdog_tiger_map:
            k, v = lis
            self.map_info['extend']['tigdog_tiger'][k] = v
            self.map_info['shrink']['tigdog_tiger'][v] = k

    def __call__(self, image, target):
        kps = target['keypoints'].copy()
        vis = target['visible'].copy()

        # src dataset -> mix dataset
        if self.extend_flag:
            num = 26
            des_kps = np.zeros((num, 2))
            des_vis = np.zeros(num)
            map_info = self.map_info['extend'][target['dataset']]
            for key in map_info:
                val = map_info[key]
                des_kps[val] = kps[key]
                des_vis[val] = vis[key]
            if np.count_nonzero(des_vis) != np.count_nonzero(vis):
                print("error vis value")

        # mix dataset -> src dataset
        else:
            if target['dataset'] == 'ap_10k':
                num = 17
            elif target['dataset'] == 'animal_pose':
                num = 20
            elif target['dataset'] == 'tigdog_horse' or target['dataset'] == 'tigdog' or target['dataset'] == 'tigdog_tiger':
                num = 19
            des_kps = np.zeros((num, 2))
            des_vis = np.zeros(num)
            map_info = self.map_info['shrink'][target['dataset']]
            for key in map_info:
                val = map_info[key]
                des_kps[val] = kps[key]
                des_vis[val] = kps[key]
        target['keypoints'] = des_kps.copy()
        target['keypoints_ori'] = des_kps.copy()
        target['visible'] = des_vis.copy()
        return image, target


class LabelFormatTransAP10KAnimalPose(object):
    def __init__(self, extend_flag=True):
        self.extend_flag = extend_flag
        self.map_info = {'extend': {'ap_10k': {}, 'animal_pose': {}},
                         'shrink': {'ap_10k': {}, 'animal_pose': {}}
                         }
        # [ap_10k,mixed]
        ap_10k_map = [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],
                    [8,8],[9,9],[10,10],[11,11],[12,12],[13,13],[14,14],[15,15],
                    [16,16]]
        animal_pose_map = [[0,0],[1,1],[2,2],[3,17],[4,18],[5,5],[6,8],[7,11],
                         [8,14],[9,6],[10,9],[11,12],[12,15],[13,7],[14,10],[15,13],
                         [16,16],[17,19],[18,20],[19,4]]

        for lis in ap_10k_map:
            k, v = lis
            self.map_info['extend']['ap_10k'][k] = v
            self.map_info['shrink']['ap_10k'][v] = k
        for lis in animal_pose_map:
            k, v = lis
            self.map_info['extend']['animal_pose'][k] = v
            self.map_info['shrink']['animal_pose'][v] = k

    def __call__(self, image, target):
        kps = target['keypoints'].copy()
        vis = target['visible'].copy()

        # src dataset -> mix dataset
        if self.extend_flag:
            num = 21
            des_kps = np.zeros((num, 2))
            des_vis = np.zeros(num)
            map_info = self.map_info['extend'][target['dataset']]
            for key in map_info:
                val = map_info[key]
                des_kps[val] = kps[key]
                des_vis[val] = vis[key]
            if np.count_nonzero(des_vis) != np.count_nonzero(vis):
                print("error vis value")

        # mix dataset -> src dataset
        else:
            if target['dataset'] == 'ap_10k':
                num = 17
            elif target['dataset'] == 'animal_pose':
                num = 20
            elif target['dataset'] == 'tigdog_horse' or target['dataset'] == 'tigdog_tiger':
                num = 19
            des_kps = np.zeros((num, 2))
            des_vis = np.zeros(num)
            map_info = self.map_info['shrink'][target['dataset']]
            for key in map_info:
                val = map_info[key]
                des_kps[val] = kps[key]
                des_vis[val] = kps[key]
        target['keypoints'] = des_kps.copy()
        target['keypoints_ori'] = des_kps.copy()
        target['visible'] = des_vis.copy()
        target['visible_ori'] = des_vis.copy()
        return image, target


# (17,3)->(51,)
class OnlyLabelFormatTrans(object):
    def __init__(self, extend_flag=True):
        self.extend_flag = extend_flag
        self.map_info = {'extend': {'ap_10k': {}, 'animal_pose': {}, "tigdog_horse":{},"tigdog_tiger":{}},
                         'shrink': {'ap_10k': {}, 'animal_pose': {}, "tigdog_horse":{},"tigdog_tiger":{}}
                         }
        # [ap_10k,mixed]
        ap_10k_map = [[0, 0], [1, 1], [2, 4], [3, 8], [4, 25], [5, 11], [6, 15], [7, 21],
                      [8, 12], [9, 16], [10, 22], [11, 13], [12, 17], [13, 23], [14, 14], [15, 18], [16, 24]]
        animal_pose_map = [[0, 0], [1, 1], [2, 4], [3, 2], [4, 3], [5, 11], [6, 12], [7, 13],
                           [8, 14], [9, 15], [10, 16], [11, 17], [12, 18], [13, 21], [14, 22], [15, 23],
                           [16, 24], [17, 6], [18, 7], [19, 25]]
        tigdog_horse_map = [[0, 0], [1, 1], [2, 5], [3, 21], [4, 22], [5, 23], [6, 24], [7, 25],
                            [8, 15], [9, 16], [10, 17], [11, 18], [12, 9], [13, 10], [14, 11], [15, 12],
                            [16, 13], [17, 14], [18, 8]]
        tigdog_tiger_map = [[0, 0], [1, 1], [2, 5], [3, 21], [4, 22], [5, 23], [6, 24], [7, 25],
                            [8, 19], [9, 20], [10, 17], [11, 18], [12, 9], [13, 10], [14, 11], [15, 12],
                            [16, 13], [17, 14], [18, 8]]
        for lis in ap_10k_map:
            k, v = lis
            self.map_info['extend']['ap_10k'][k] = v
            self.map_info['shrink']['ap_10k'][v] = k
        for lis in animal_pose_map:
            k, v = lis
            self.map_info['extend']['animal_pose'][k] = v
            self.map_info['shrink']['animal_pose'][v] = k
        for lis in tigdog_horse_map:
            k, v = lis
            self.map_info['extend']['tigdog_horse'][k] = v
            self.map_info['shrink']['tigdog_horse'][v] = k
        for lis in tigdog_tiger_map:
            k, v = lis
            self.map_info['extend']['tigdog_tiger'][k] = v
            self.map_info['shrink']['tigdog_tiger'][v] = k

    # (26,3) -> (17,3) -> (51,)
    def __call__(self, target):
        kps = np.array(target['keypoints'].copy()).reshape(-1,3)
        vis = kps[:,-1].copy()
        kps = kps[:,:-1].copy()
        # src dataset -> mix dataset
        if self.extend_flag:
            num = 26
            des_kps = np.zeros((num, 2))
            des_vis = np.zeros(num)
            map_info = self.map_info['extend'][target['dataset']]
            for key in map_info:
                val = map_info[key]
                des_kps[val] = kps[key]
                des_vis[val] = vis[key]
            target['keypoints'] = des_kps
            target['visible'] = des_vis
        # mix dataset -> src dataset
        else:
            if target['dataset'] == 'ap_10k':
                num = 17
            elif target['dataset'] == 'animal_pose':
                num = 20
            elif target['dataset'] == 'tigdog_horse' or target['dataset'] == 'tigdog_tiger':
                num = 19
            des_kps = np.zeros((num, 2))
            des_vis = np.zeros(num)
            map_info = self.map_info['shrink'][target['dataset']]
            for key in map_info:
                val = map_info[key]
                des_kps[val] = kps[key]
                des_vis[val] = vis[key]
            combine_array = np.column_stack((des_kps, des_vis))
            target['keypoints'] = combine_array.flatten().tolist()
            # target['visible'] = des_vis.tolist()
        return target


class OnlyLabelFormatTransAP10KAnimalPose(object):
    def __init__(self, extend_flag=True):
        self.extend_flag = extend_flag
        self.map_info = {'extend': {'ap_10k': {}, 'animal_pose': {}},
                         'shrink': {'ap_10k': {}, 'animal_pose': {}}
                         }
        # [ap_10k,mixed]
        ap_10k_map = [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],
                    [8,8],[9,9],[10,10],[11,11],[12,12],[13,13],[14,14],[15,15],
                    [16,16]]
        animal_pose_map = [[0,0],[1,1],[2,2],[3,17],[4,18],[5,5],[6,8],[7,11],
                         [8,14],[9,6],[10,9],[11,12],[12,15],[13,7],[14,10],[15,13],
                         [16,16],[17,19],[18,20],[19,4]]
        for lis in ap_10k_map:
            k, v = lis
            self.map_info['extend']['ap_10k'][k] = v
            self.map_info['shrink']['ap_10k'][v] = k
        for lis in animal_pose_map:
            k, v = lis
            self.map_info['extend']['animal_pose'][k] = v
            self.map_info['shrink']['animal_pose'][v] = k

    # (26,3) -> (17,3) -> (51,)
    def __call__(self, target):
        kps = np.array(target['keypoints'].copy()).reshape(-1,3)
        vis = kps[:,-1].copy()
        kps = kps[:,:-1].copy()
        # src dataset -> mix dataset
        if self.extend_flag:
            num = 21
            des_kps = np.zeros((num, 2))
            des_vis = np.zeros(num)
            map_info = self.map_info['extend'][target['dataset']]
            for key in map_info:
                val = map_info[key]
                des_kps[val] = kps[key]
                des_vis[val] = vis[key]
            target['keypoints'] = des_kps
            target['visible'] = des_vis
        # mix dataset -> src dataset
        else:
            if target['dataset'] == 'ap_10k':
                num = 17
            elif target['dataset'] == 'animal_pose':
                num = 20
            elif target['dataset'] == 'tigdog_horse' or target['dataset'] == 'tigdog_tiger':
                num = 19
            des_kps = np.zeros((num, 2))
            des_vis = np.zeros(num)
            map_info = self.map_info['shrink'][target['dataset']]
            for key in map_info:
                val = map_info[key]
                des_kps[val] = kps[key]
                des_vis[val] = vis[key]
            combine_array = np.column_stack((des_kps, des_vis))
            target['keypoints'] = combine_array.flatten().tolist()
            # target['visible'] = des_vis.tolist()
        return target


# (17,2)->(26,2)
# (17)->(26)
class OriginalLabelFormatTrans(object):
    def __init__(self, extend_flag=True):
        self.extend_flag = extend_flag
        self.map_info = {'extend': {'ap_10k': {}, 'animal_pose': {}, "tigdog_horse":{},"tigdog_tiger":{}},
                         'shrink': {'ap_10k': {}, 'animal_pose': {}, "tigdog_horse":{},"tigdog_tiger":{}}
                         }
        # [ap_10k,mixed]
        ap_10k_map = [[0, 0], [1, 1], [2, 4], [3, 8], [4, 25], [5, 11], [6, 15], [7, 21],
                      [8, 12], [9, 16], [10, 22], [11, 13], [12, 17], [13, 23], [14, 14], [15, 18], [16, 24]]
        animal_pose_map = [[0, 0], [1, 1], [2, 4], [3, 2], [4, 3], [5, 11], [6, 12], [7, 13],
                           [8, 14], [9, 15], [10, 16], [11, 17], [12, 18], [13, 21], [14, 22], [15, 23],
                           [16, 24], [17, 6], [18, 7], [19, 25]]
        tigdog_horse_map = [[0, 0], [1, 1], [2, 5], [3, 21], [4, 22], [5, 23], [6, 24], [7, 25],
                            [8, 15], [9, 16], [10, 17], [11, 18], [12, 9], [13, 10], [14, 11], [15, 12],
                            [16, 13], [17, 14], [18, 8]]
        tigdog_tiger_map = [[0, 0], [1, 1], [2, 5], [3, 21], [4, 22], [5, 23], [6, 24], [7, 25],
                            [8, 19], [9, 20], [10, 17], [11, 18], [12, 9], [13, 10], [14, 11], [15, 12],
                            [16, 13], [17, 14], [18, 8]]
        for lis in ap_10k_map:
            k, v = lis
            self.map_info['extend']['ap_10k'][k] = v
            self.map_info['shrink']['ap_10k'][v] = k
        for lis in animal_pose_map:
            k, v = lis
            self.map_info['extend']['animal_pose'][k] = v
            self.map_info['shrink']['animal_pose'][v] = k
        for lis in tigdog_horse_map:
            k, v = lis
            self.map_info['extend']['tigdog_horse'][k] = v
            self.map_info['shrink']['tigdog_horse'][v] = k
        for lis in tigdog_tiger_map:
            k, v = lis
            self.map_info['extend']['tigdog_tiger'][k] = v
            self.map_info['shrink']['tigdog_tiger'][v] = k

    # (17,2)->(26,2)
    # (17)->(26)
    def __call__(self, target):
        vis = target['visible'].copy()
        kps = target['keypoints'].copy()
        # src dataset -> mix dataset
        if self.extend_flag:
            num = 26
            des_kps = np.zeros((num, 2))
            des_vis = np.zeros(num)
            map_info = self.map_info['extend'][target['dataset']]
            for key in map_info:
                val = map_info[key]
                des_kps[val] = kps[key]
                des_vis[val] = vis[key]
            target['keypoints'] = des_kps
            target['visible'] = des_vis
        # mix dataset -> src dataset
        else:
            if target['dataset'] == 'ap_10k':
                num = 17
            elif target['dataset'] == 'animal_pose':
                num = 20
            elif target['dataset'] == 'tigdog_horse' or target['dataset'] == 'tigdog_tiger':
                num = 19
            des_kps = np.zeros((num, 2))
            des_vis = np.zeros(num)
            map_info = self.map_info['shrink'][target['dataset']]
            for key in map_info:
                val = map_info[key]
                des_kps[val] = kps[key]
                des_vis[val] = vis[key]
            combine_array = np.column_stack((des_kps, des_vis))
            target['keypoints'] = combine_array.flatten().tolist()
            target['visible'] = des_vis.tolist()
        return target


class OriginalLabelFormatTransAP10KAnimalPose(object):
    def __init__(self, extend_flag=True):
        self.extend_flag = extend_flag
        self.map_info = {'extend': {'ap_10k': {}, 'animal_pose': {}},
                         'shrink': {'ap_10k': {}, 'animal_pose': {}}
                         }
        # [ap_10k,mixed]
        ap_10k_map = [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],
                    [8,8],[9,9],[10,10],[11,11],[12,12],[13,13],[14,14],[15,15],
                    [16,16]]
        animal_pose_map = [[0,0],[1,1],[2,2],[3,17],[4,18],[5,5],[6,8],[7,11],
                         [8,14],[9,6],[10,9],[11,12],[12,15],[13,7],[14,10],[15,13],
                         [16,16],[17,19],[18,20],[19,4]]
        for lis in ap_10k_map:
            k, v = lis
            self.map_info['extend']['ap_10k'][k] = v
            self.map_info['shrink']['ap_10k'][v] = k
        for lis in animal_pose_map:
            k, v = lis
            self.map_info['extend']['animal_pose'][k] = v
            self.map_info['shrink']['animal_pose'][v] = k

    # (17,2)->(26,2)
    # (17)->(26)
    def __call__(self, target):
        vis = target['visible'].copy()
        kps = target['keypoints'].copy()
        # src dataset -> mix dataset
        if self.extend_flag:
            num = 21
            des_kps = np.zeros((num, 2))
            des_vis = np.zeros(num)
            map_info = self.map_info['extend'][target['dataset']]
            for key in map_info:
                val = map_info[key]
                des_kps[val] = kps[key]
                des_vis[val] = vis[key]
            target['keypoints'] = des_kps
            target['visible'] = des_vis
        # mix dataset -> src dataset
        else:
            if target['dataset'] == 'ap_10k':
                num = 17
            elif target['dataset'] == 'animal_pose':
                num = 20
            else:
                return
            des_kps = np.zeros((num, 2))
            des_vis = np.zeros(num)
            map_info = self.map_info['shrink'][target['dataset']]
            for key in map_info:
                val = map_info[key]
                des_kps[val] = kps[key]
                des_vis[val] = vis[key]
            combine_array = np.column_stack((des_kps, des_vis))
            target['keypoints'] = combine_array.flatten().tolist()
            target['visible'] = des_vis.tolist()
        return target


# 适用于没有标签文件 只有可见性的情况
class NoLabelFormatTrans(object):
    def __init__(self, extend_flag=True):
        self.extend_flag = extend_flag
        self.map_info = {'extend': {'ap_10k': {}, 'animal_pose': {}, "tigdog_horse":{},"tigdog_tiger":{}},
                         'shrink': {'ap_10k': {}, 'animal_pose': {}, "tigdog_horse":{},"tigdog_tiger":{}}
                         }
        # [ap_10k,mixed]
        ap_10k_map = [[0, 0], [1, 1], [2, 4], [3, 8], [4, 25], [5, 11], [6, 15], [7, 21],
                      [8, 12], [9, 16], [10, 22], [11, 13], [12, 17], [13, 23], [14, 14], [15, 18], [16, 24]]
        animal_pose_map = [[0, 0], [1, 1], [2, 4], [3, 2], [4, 3], [5, 11], [6, 12], [7, 13],
                           [8, 14], [9, 15], [10, 16], [11, 17], [12, 18], [13, 21], [14, 22], [15, 23],
                           [16, 24], [17, 6], [18, 7], [19, 25]]
        tigdog_horse_map = [[0, 0], [1, 1], [2, 5], [3, 21], [4, 22], [5, 23], [6, 24], [7, 25],
                            [8, 15], [9, 16], [10, 17], [11, 18], [12, 9], [13, 10], [14, 11], [15, 12],
                            [16, 13], [17, 14], [18, 8]]
        tigdog_tiger_map = [[0, 0], [1, 1], [2, 5], [3, 21], [4, 22], [5, 23], [6, 24], [7, 25],
                            [8, 19], [9, 20], [10, 17], [11, 18], [12, 9], [13, 10], [14, 11], [15, 12],
                            [16, 13], [17, 14], [18, 8]]
        for lis in ap_10k_map:
            k, v = lis
            self.map_info['extend']['ap_10k'][k] = v
            self.map_info['shrink']['ap_10k'][v] = k
        for lis in animal_pose_map:
            k, v = lis
            self.map_info['extend']['animal_pose'][k] = v
            self.map_info['shrink']['animal_pose'][v] = k
        for lis in tigdog_horse_map:
            k, v = lis
            self.map_info['extend']['tigdog_horse'][k] = v
            self.map_info['shrink']['tigdog_horse'][v] = k
        for lis in tigdog_tiger_map:
            k, v = lis
            self.map_info['extend']['tigdog_tiger'][k] = v
            self.map_info['shrink']['tigdog_tiger'][v] = k

    # (17,2)->(26,2)
    # (17)->(26)
    def __call__(self, target_vis,dataset):
        vis = target_vis.copy()
        # src dataset -> mix dataset
        if self.extend_flag:
            num = 26
            des_vis = np.zeros(num)
            map_info = self.map_info['extend'][dataset]
            for key in map_info:
                val = map_info[key]
                des_vis[val] = vis[key]
            vis = des_vis
        # mix dataset -> src dataset
        else:
            if dataset == 'ap_10k':
                num = 17
            elif dataset == 'animal_pose':
                num = 20
            elif dataset == 'tigdog_horse' or dataset == 'tigdog_tiger':
                num = 19
            des_vis = np.zeros(num)
            map_info = self.map_info['shrink'][dataset]
            for key in map_info:
                val = map_info[key]
                des_vis[val] = vis[key]
            vis = des_vis.tolist()
        return vis


class TransformMPL(object):
    def __init__(self, args, mean, std,n=2,m=10):
        with open(args.keypoints_path, "r") as f:
            animal_kps_info = json.load(f)
        kps_weights = np.array(animal_kps_info["kps_weights"],dtype=np.float32).reshape((args.num_joints,))

        self.ori = Compose([
            HalfBody(0.3, animal_kps_info["upper_body_ids"], animal_kps_info["lower_body_ids"]),
            AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=(256,256)),
            RandomHorizontalFlip(0.5, animal_kps_info["flip_pairs"]),
            KeypointToHeatMap(heatmap_hw=(64,64), gaussian_sigma=2, keypoints_weights=kps_weights)]
        )
        self.aug = Compose([
            RandWeakAugment(n,m)]
        )
        self.normalize = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)])

    def __call__(self, img,target):
        ori_img,ori_target = self.ori(img,target)
        aug_img,aug_target = self.aug(ori_img,ori_target)
        ori_img,ori_target = self.normalize(ori_img,ori_target)
        aug_img,aug_target = self.normalize(aug_img,aug_target)

        return [ori_img,aug_img], ori_target


class TransformConsistency(object):
    def __init__(self, args, mean, std,n=2,m=10):
        weak_scale = 0.2
        weak_rot = 30
        strong_scale = 0.35
        strong_rot = 45

        # flag for halfbody and flip

        with open(args.keypoints_path, "r") as f:
            animal_kps_info = json.load(f)
        kps_weights = np.array(animal_kps_info["kps_weights"],dtype=np.float32).reshape((args.num_joints,))

        self.weak = Compose([
            # HalfBody(0.3, animal_kps_info["upper_body_ids"], animal_kps_info["lower_body_ids"]),
            AffineTransform(scale=(1 - weak_scale, 1 + weak_scale), rotation=(-weak_rot, weak_rot), fixed_size=(256,256)),
            # RandomHorizontalFlip(0.5, animal_kps_info["flip_pairs"]),
            KeypointToHeatMap(heatmap_hw=(64,64), gaussian_sigma=2, keypoints_weights=kps_weights)]
        )
        self.strong = Compose([
            # HalfBody(0.3, animal_kps_info["upper_body_ids"], animal_kps_info["lower_body_ids"]),
            AffineTransform(scale=(1 - strong_scale,1 + strong_scale), rotation=(-strong_rot, strong_rot),
                            fixed_size=(256, 256)),
            # RandomHorizontalFlip(0.5, animal_kps_info["flip_pairs"]),
            KeypointToHeatMap(heatmap_hw=(64, 64), gaussian_sigma=2, keypoints_weights=kps_weights),
            RandWeakAugment(n,m)]
        )
        self.normalize = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)])

    def __call__(self, img,target):
        weak_img,weak_target = self.weak(img,copy.deepcopy(target))
        strong_img,strong_target = self.strong(img,copy.deepcopy(target))
        weak_img,weak_target = self.normalize(weak_img,weak_target)
        strong_img,strong_target = self.normalize(strong_img,strong_target)

        return [weak_img,strong_img], [weak_target,strong_target]


class TransformFixMatch(object):
    def __init__(self, args, mean, std,n=2,m=10):
        with open(args.keypoints_path, "r") as f:
            animal_kps_info = json.load(f)
        kps_weights = np.array(animal_kps_info["kps_weights"],dtype=np.float32).reshape((args.num_joints,))

        self.ori = Compose([
            HalfBody(0.3, animal_kps_info["upper_body_ids"], animal_kps_info["lower_body_ids"]),
            AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=(256,256)),
            RandomHorizontalFlip(0.5, animal_kps_info["flip_pairs"]),
            KeypointToHeatMap(heatmap_hw=(64,64), gaussian_sigma=2, keypoints_weights=kps_weights)]
        )
        self.aug = Compose([
            RandWeakAugmentFixMatch(n,m)]
        )
        self.normalize = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)])

    def __call__(self, img,target):
        ori_img,ori_target = self.ori(img,target)
        aug_img,aug_target = self.aug(ori_img,ori_target)
        ori_img,ori_target = self.normalize(ori_img,ori_target)
        aug_img,aug_target = self.normalize(aug_img,aug_target)

        return [ori_img,aug_img], ori_target


if __name__ == '__main__':
    # ap_10k_map = [[0, 0], [1, 1], [2, 4], [3, 8], [4, 25], [5, 11], [6, 15], [7, 21],
    #               [8, 12], [9, 16], [10, 22], [11, 13], [12, 17], [13, 23], [14, 14], [15, 18], [16, 24]]
    # animal_pose_map = [[0, 0], [1, 1], [2, 4], [3, 2], [4, 3], [5, 11], [6, 12], [7, 13],
    #                    [8, 14], [9, 15], [10, 16], [11, 17], [12, 18], [13, 21], [14, 22], [15, 23],
    #                    [16, 24], [17, 6], [18, 7], [19, 25]]
    # tigdog_map = [[0, 0], [1, 1], [2, 5], [3, 21], [4, 22], [5, 23], [6, 24], [7, 25],
    #               [8, 15], [9, 16], [10, 17], [11, 18], [12, 9], [13, 10], [14, 11], [15, 12],
    #               [16, 13], [17, 14], [18, 8]]
    # with open('./info/ap_10k_keypoints_format.json','r') as f:
    #     ap_10k_data = json.load(f)
    # with open('./info/animal_pose_keypoints_format.json','r') as f:
    #     animal_pose_data = json.load(f)
    # with open('./info/tigdog_keypoints_format.json','r') as f:
    #     tigdog_data = json.load(f)
    # with open('./info/keypoints_definition.json','r') as f:
    #     mix_data = json.load(f)
    #
    # for k,v in ap_10k_map:
    #     key_1 = ap_10k_data['keypoints'][k]
    #     key_2 = mix_data['keypoints'][v]
    #     print(f"{key_1} : {key_2}")
    # for k,v in animal_pose_map:
    #     key_1 = animal_pose_data['keypoints'][k]
    #     key_2 = mix_data['keypoints'][v]
    #     print(f"{key_1} : {key_2}")
    # for k,v in tigdog_map:
    #     key_1 = tigdog_data['keypoints'][k]
    #     key_2 = mix_data['keypoints'][v]
    #     print(f"{key_1} : {key_2}")

    func = LabelFormatTrans(dataset='ap_10k', extend_flag=True)
    print('pause')
