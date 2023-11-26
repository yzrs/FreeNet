import random
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import train_utils.transforms
from models.hrnet import HighResolutionNet
import torchvision.transforms as transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inference():
    set_seed(2)

    # Ls / + Lu / +Lf
    weights_path = "saved_weights/ls_lu_lf.pth"
    model = HighResolutionNet(num_joints=21)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    model.to("cuda:0")

    image = cv2.imread('test.jpg')

    # 将图像调整为固定大小
    desired_size = (256, 256)  # 设定目标大小
    resized_image = cv2.resize(image, desired_size)

    # 将图像转换为张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    normalized_image = transform(resized_image)

    with torch.no_grad():
        normalized_image = normalized_image.unsqueeze(0).to("cuda:0")
        outputs = model(normalized_image)
        coords,confidence = train_utils.transforms.get_max_preds(outputs)
        confidence = np.array(confidence.squeeze(0).squeeze(-1).cpu())

        # 反归一化图像tensor
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img_tensor = (normalized_image.cpu() * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)) * 255.0
        # 将图像tensor转换为numpy数组
        img_np = img_tensor.squeeze(0).numpy().transpose(1, 2, 0).astype(int)

        # 缩放关键点坐标至目标尺寸
        source_size = torch.tensor([256, 256])
        target_size = torch.tensor([64, 64])
        scaled_keypoints_tensor = coords.cpu().squeeze(0) * (source_size.float() / target_size.float())
        scaled_keypoints = scaled_keypoints_tensor.int().tolist()
        for i,point in enumerate(scaled_keypoints):
            x, y = point
            if confidence[i] > 0.4:
                plt.scatter(x, y, s=100,color='white',edgecolors='black')  # 绘制散点图

        # 显示图像
        plt.imshow(img_np)
        plt.axis('off')  # 可选，关闭坐标轴
        plt.show()


if __name__ == '__main__':
    inference()
