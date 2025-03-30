#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# from skimage.measure import compare_ssim, compare_psnr, compare_mse
import cv2
#from skimage import measure
# from skimage.metrics import structural_similarity as compare_ssim
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

class NormalLoss(nn.Module):
    def __init__(self,ignore_lb=255, *args, **kwargs):
        super( NormalLoss, self).__init__()
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels)
        return torch.mean(loss)

from math import exp

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):

    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # import ipdb
    # ipdb.set_trace()
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    # import ipdb
    # ipdb.set_trace()
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel) # 高斯滤波 求均值
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel) # 求均值

    mu1_sq = mu1.pow(2) # 平方
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq # var(x) = Var(X)=E[X^2]-E[X]^2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2 # 协方差

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img,i):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        # cv.imshow("y_grad", y_grad[0,0,...].cpu().numpy())
        ir_grad=self.sobelconv(image_ir)
        # cv.imshow("ir_grad", ir_grad[0,0,...].cpu().numpy())
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        # x_grad_joint=ir_grad #下雨天的可见光边界信息都是雨的轮廓
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_ssim=1-(ssim(image_y, generate_img))/2-(ssim(image_ir, generate_img))/2
        loss_total=loss_in+10*loss_grad+loss_ssim
        return loss_total,loss_in,loss_grad,loss_ssim

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)





def getSSIM(img1, img2):
    return compare_ssim(img1, img2, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True


def getPSNR(img1, img2):
    return compare_psnr(img1, img2)


def getMSE(img1, img2):
    return compare_mse(img1, img2)


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def smooth(image, sigma=1.4, length=5):
    """ Smooth the image
    Compute a gaussian filter with sigma = sigma and kernal_length = length.
    Each element in the kernal can be computed as below:
        G[i, j] = (1/(2*pi*sigma**2))*exp(-((i-k-1)**2 + (j-k-1)**2)/2*sigma**2)
    Then, use the gaussian filter to smooth the input image.

    Args:
        image: array of grey image
        sigma: the sigma of gaussian filter, default to be 1.4
        length: the kernal length, default to be 5

    Returns:
        the smoothed image
    """
    # Compute gaussian filter
    k = length // 2
    gaussian = np.zeros([length, length])
    for i in range(length):
        for j in range(length):
            gaussian[i, j] = np.exp(-((i - k) ** 2 + (j - k) ** 2) / (2 * sigma ** 2))
    gaussian /= 2 * np.pi * sigma ** 2
    # Batch Normalization
    gaussian = gaussian / np.sum(gaussian)

    # Use Gaussian Filter
    W, H = image.shape
    new_image = np.zeros([W - k * 2, H - k * 2])

    for i in range(W - 2 * k):
        for j in range(H - 2 * k):
            # 卷积运算
            new_image[i, j] = np.sum(image[i:i + length, j:j + length] * gaussian)

    new_image = np.uint8(new_image)
    return new_image


def get_gradient_and_direction(image):
    """ Compute gradients and its direction
    Use Sobel filter to compute gradients and direction.
         -1 0 1        -1 -2 -1
    Gx = -2 0 2   Gy =  0  0  0
         -1 0 1         1  2  1

    Args:
        image: array of grey image

    Returns:
        gradients: the gradients of each pixel
        direction: the direction of the gradients of each pixel
    """
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    W, H = image.shape
    gradients = np.zeros([W - 2, H - 2])
    direction = np.zeros([W - 2, H - 2])

    for i in range(W - 2):
        for j in range(H - 2):
            dx = np.sum(image[i:i + 3, j:j + 3] * Gx)
            dy = np.sum(image[i:i + 3, j:j + 3] * Gy)
            gradients[i, j] = np.sqrt(dx ** 2 + dy ** 2)
            if dx == 0:
                direction[i, j] = np.pi / 2
            else:
                direction[i, j] = np.arctan(dy / dx)

    gradients = np.uint8(gradients)
    return gradients, direction


def NMS(gradients, direction):
    """ Non-maxima suppression
    Args:
        gradients: the gradients of each pixel
        direction: the direction of the gradients of each pixel

    Returns:
        the output image
    """
    W, H = gradients.shape
    nms = np.copy(gradients[1:-1, 1:-1])

    for i in range(1, W - 1):
        for j in range(1, H - 1):
            theta = direction[i, j]
            weight = np.tan(theta)
            if theta > np.pi / 4:
                d1 = [0, 1]
                d2 = [1, 1]
                weight = 1 / weight
            elif theta >= 0:
                d1 = [1, 0]
                d2 = [1, 1]
            elif theta >= - np.pi / 4:
                d1 = [1, 0]
                d2 = [1, -1]
                weight *= -1
            else:
                d1 = [0, -1]
                d2 = [1, -1]
                weight = -1 / weight

            g1 = gradients[i + d1[0], j + d1[1]]
            g2 = gradients[i + d2[0], j + d2[1]]
            g3 = gradients[i - d1[0], j - d1[1]]
            g4 = gradients[i - d2[0], j - d2[1]]

            grade_count1 = g1 * weight + g2 * (1 - weight)
            grade_count2 = g3 * weight + g4 * (1 - weight)

            if grade_count1 > gradients[i, j] or grade_count2 > gradients[i, j]:
                nms[i - 1, j - 1] = 0
    return nms


def double_threshold(nms, threshold1, threshold2):
    """ Double Threshold
    Use two thresholds to compute the edge.

    Args:
        nms: the input image
        threshold1: the low threshold
        threshold2: the high threshold

    Returns:
        The binary image.
    """
    visited = np.zeros_like(nms)
    output_image = nms.copy()
    W, H = output_image.shape

    def dfs(i, j):
        if i >= W or i < 0 or j >= H or j < 0 or visited[i, j] == 1:
            return
        visited[i, j] = 1
        if output_image[i, j] > threshold1:
            output_image[i, j] = 255
            dfs(i - 1, j - 1)
            dfs(i - 1, j)
            dfs(i - 1, j + 1)
            dfs(i, j - 1)
            dfs(i, j + 1)
            dfs(i + 1, j - 1)
            dfs(i + 1, j)
            dfs(i + 1, j + 1)
        else:
            output_image[i, j] = 0

    for w in range(W):
        for h in range(H):
            if visited[w, h] == 1:
                continue
            if output_image[w, h] >= threshold2:
                dfs(w, h)
            elif output_image[w, h] <= threshold1:
                output_image[w, h] = 0
                visited[w, h] = 1

    for w in range(W):
        for h in range(H):
            if visited[w, h] == 0:
                output_image[w, h] = 0
    return output_image



if __name__ == '__main__':
    img1 = cv2.imread('D:/new_airport/JPEGImages/random_level/airsport_rainfall_random/000006.jpg',0)
    # img2 = cv2.imread('D:/DL/tan/SeAFusion-main-0113/SeAFusion-main/MSRS/Fusion/train/M3FD/00002.png')

    cv.imshow("Original", img1)
    smoothed_image = smooth(img1)
    cv.imshow("GaussinSmooth(5*5)", smoothed_image)
    gradients, direction = get_gradient_and_direction(smoothed_image)
    # print(gradients)
    # print(direction)
    nms = NMS(gradients, direction)
    output_image = double_threshold(nms, 40, 100)
    cv.imshow("outputImage", output_image)
    cv.waitKey(0)

    # ssim=getSSIM(img1, img2)
    # PSNR=getPSNR(img1, img2)
    # MSE=getMSE(img1, img2)



