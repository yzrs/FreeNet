import logging
import random
import cv2
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt


logger = logging.getLogger(__name__)

PARAMETER_MAX = 10
RESAMPLE_MODE = Image.BICUBIC
FILL_COLOR = (128, 128, 128)


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = 8 - _round_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Sharpness(img, v, max_v, bias):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Solarize(img, v, max_v, **kwarg):
    v = _int_parameter(v, max_v)
    return PIL.ImageOps.solarize(img, 255 - v)


# 根据图像大小的百分比erase图像
def Cutout(img, v, max_v, **kwarg):
    if v == 0:
        return img
    v = _float_parameter(v, max_v)
    v = int(v * min(img.size))

    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = FILL_COLOR
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def CutoutMore(img, v, max_v, **kwarg):
    if v == 0:
        return img
    erase_times = random.randint(1,5)
    for _ in range(erase_times):
        ratio = _float_parameter(v, max_v)
        cut_size = int(ratio * min(img.size[0], img.size[1]))
        w, h = img.size
        x0 = np.random.uniform(0, w)
        y0 = np.random.uniform(0, h)
        x0 = int(max(0, x0 - cut_size / 2.))
        y0 = int(max(0, y0 - cut_size / 2.))
        x1 = int(min(w, x0 + cut_size))
        y1 = int(min(h, y0 + cut_size))
        xy = (x0, y0, x1, y1)
        # gray
        color = FILL_COLOR
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def apply_blur(image, radius):
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return blurred_image


def blur(img, v, max_v, blur_radius=3, **kwargs):
    if v == 0:
        return img
    img = apply_blur(img,blur_radius)

    return img


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def _round_parameter(v, max_v):
    return int(round(v * max_v / PARAMETER_MAX))


# AutoContrast 直方图均衡化
# Equalize 对图像的亮度通道进行直方图均衡化 ***
# Invert 颜色反转  ****
# Posterize：减少图像中每个像素的颜色位数，从而产生色调分离的效果。
# Solarize：对图像中亮度高于阈值的像素进行颜色反转，从而产生曝光过度的效果。 ****
# Contrast：调整图像的对比度。 v>1 增强对比度  v<1 减少对比度 ****
# Brightness：调整图像的亮度。  ***
# Sharpness：调整图像的锐度。
def color_augment_pool():
    augs = [
        (AutoContrast, None, None),
        (Equalize, None, None),
        (Invert, None, None),
        (Color, 1.8, 0.1),
        (Posterize, 4, 0),
        (Solarize, 256, None),
        (Contrast, 1.8, 0.1),
        (Brightness, 1.8, 0.1),
        (Sharpness, 1.8, 0.1)
    ]
    return augs


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [
        (AutoContrast, None, None),
        (Brightness, 0.9, 0.05),
        (Color, 0.9, 0.05),
        (Contrast, 0.9, 0.05),
        (Equalize, None, None),
        (Identity, None, None),
        (Posterize, 4, 4),
        (Sharpness, 0.9, 0.05),
        (Solarize, 256, None),
    ]
    return augs


def color_augment_pool_test():
    augs = [
        # color jitter
        (AutoContrast, None, None),
        (Brightness, 0.9, 0.05),
        (Color, 0.9, 0.05),
        (Contrast, 0.9, 0.05),
        (Equalize, None, None),
        (Identity, None, None),
        (Posterize, 4, 4),
        (Sharpness, 0.9, 0.05),
        (Solarize, 256, None),
        # erase
        # (Cutout,0.05,None),
        (CutoutMore,0.05,None),
        # blur
        (blur,0.1,None)
    ]
    return augs


def color_augment_pool_erase():
    augs = [
        # erase
        # (Cutout,0.05,None),
        (CutoutMore,0.05,None)
    ]
    return augs


def color_augment_pool_blur():
    augs = [
        # blur
        (blur, 0.1, None)
    ]
    return augs


class RandAugmentAP10KOneLeast(object):
    def __init__(self, n, m):
        self.n = int(n)
        self.m = m
        self.augment_pool = color_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        applied_ops = 0
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() <= prob:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
                applied_ops += 1
        if applied_ops == 0:
            op, max_v, bias = random.choice(self.augment_pool)
            img = op(img, v=self.m, max_v=max_v, bias=bias)
        return img


class RandWeakAugment(object):
    def __init__(self, n=2, m=10):
        self.n = int(n)
        self.m = m
        self.augment_pool = color_augment_pool_test()

    def __call__(self, img,target):
        img = Image.fromarray(img)
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() <= prob:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        return img,target


class RandWeakAugmentFixMatch(object):
    def __init__(self, n=2, m=10):
        self.n = int(n)
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img,target):
        img = Image.fromarray(img)
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() <= prob:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        return img,target


class TestAugmentAP10K(object):
    def __init__(self, n, m):
        self.n = int(n)
        self.m = m
        self.augment_pool = color_augment_pool_test()

    def __call__(self, img):
        results = []
        # results.append(img)
        for op, max_v, bias in self.augment_pool:
            img_aug = op(img.copy(), v=self.m, max_v=max_v, bias=bias)
            results.append(img_aug)
        return results


if __name__ == '__main__':
    img = Image.open('image.jpg')
    augmenter = TestAugmentAP10K(2, 10)
    results = augmenter(img)
    print(type(results))

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 4)
    # titles = ['origin','AutoContrast', 'Equalize', 'Invert', 'Posterize', 'Solarize', 'Contrast', 'Brightness', 'Sharpness']
    titles = ['AutoContrast', 'Equalize', 'Invert', 'color','Posterize', 'Solarize', 'Contrast', 'Brightness', 'Sharpness','Cutout','CutoutMore','blur']

    for i in range(3):
        for j in range(4):
            axs[i, j].imshow(results[i * 4 + j])
            axs[i, j].set_title(titles[i * 4 + j])
    plt.show()
