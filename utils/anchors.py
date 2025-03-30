# -*- coding: utf-8 -*-
# @Author  : LG
from torch import nn
import numpy as np
import torch
from utils.Boxs_op import corner_form_to_center_form, center_form_to_corner_form
import config.config_voc as cfg

class Anchors_5:
    """
    Retainnet anchors, 生成策略与SSD不同
    """
    def __init__(self):
        # self.features_maps = [(56, 56 ), (28, 28), (14, 14)]

        # self.anchor_wh = np.array([[(0.00347222, 0.01666667), (0.01145833, 0.01666667), (0.03177083, 0.02592593)],
        #                              [(0.04930556, 0.05555556),(0.01770833, 0.05833333), (0.10885417, 0.09907407)] ,
        #                              [(0.18958333, 0.15092593), (0.35260417, 0.30595238), (0.396875, 0.46388889)]])\
        if cfg.data_type == 'voc':
            self.anchor_wh = np.array([[(0.034,0.064 ), (0.086,0.08 ), (0.054,0.15466667), (0.134,0.14578005),(0.088,0.28)],
                                      [(0.168,0.26933333),(0.156, 0.496 ), (0.156,0.496 ),(0.294,0.19393939),(0.276, 0.66966967)] ,
                                      [(0.416,0.47733333), (0.5, 0.7957958), (0.628,0.32266667),(0.788,0.58133333),(0.896,0.898   )]])

        elif cfg.data_type == 'airport':
            self.anchor_wh = np.array([[(0.00347222, 0.01666667), (0.0109375,  0.01574074), (0.01458333, 0.03240741),(0.02916667, 0.0212963)],
                                      [(0.04270833, 0.03518519), (0.03194444, 0.06296296),(0.06302083, 0.05555556),(0.01666667, 0.06018519)],
                                      [(0.07152778, 0.12314815),(0.09479167, 0.2422619),(0.12395833, 0.10185185), (0.09114583, 0.08518519)],
                                      [(0.38125,    0.3037037),(0.22552083, 0.225),(0.203125,   0.14074074), (0.15486111, 0.1462963)] ,
                                      [(0.35208333, 0.5787037), (0.39427083, 0.46481481), (0.40833333, 0.37222222),(0.37760417, 0.57037037)]])

            # self.anchor_wh = np.array([[(0.00347222, 0.01666667), (0.01145833, 0.01574074), (0.015625, 0.05277778),
            #                             (0.028125, 0.0212963), (0.0296875, 0.06481481)],
            #                            [(0.04270833, 0.03518519), (0.059375, 0.05925926), (0.09114583, 0.08518519),
            #                             (0.09479167, 0.1462963), (0.12447917, 0.10092593)],
            #                            [(0.3859375, 0.31296296), (0.39322917, 0.46574074), (0.22552083, 0.225),
            #                             (0.203125, 0.14074074), (0.15572917, 0.14537037)]])
        elif cfg.data_type == 'kitti':
            self.anchor_wh = np.array([[(0.0273752,0.216), (0.0410628,0.10666667), (0.08615137,0.2), (0.05716586,0.14666667),(0.01449275,0.11621622)],
                                      [(0.10708535,0.13333333),(0.1763285,0.40533333), (0.26731079,0.48266667),(0.14251208,0.24533333),(0.05877617,0.42933333)] ,
                                      [(0.04991948,0.06133333), (0.07246377,0.09333333), (0.02254428,0.05066667),(0.02979066,0.07466667),(0.01207729,0.03466667)]])
        # self.anchor_wh = np.array([[(0.00347222, 0.01666667),(0.01770833, 0.06666667), (0.01145833, 0.01574074), (0.01458333, 0.04166667),(0.04270833, 0.03518519),(0.02777778, 0.05648148),(0.02847222, 0.0212963),(0.04010417, 0.06309524)],
        #                           [(0.05885417, 0.06851852),(0.0791666, 0.05277778),(0.09114583, 0.08518519),(0.0947916, 0.1797619),(0.12291667, 0.09722222),(0.12708333, 0.12685185), (0.17881944, 0.225  ),(0.15694444, 0.15)] ,
        #                           [(0.19947917, 0.13888889), (0.18020833, 0.30595238), (0.28072917, 0.38703704),(0.22552083, 0.19814815),(0.3859375,  0.30740741),(0.390625,   0.46944444),(0.30052083, 0.26574074),(0.41666667, 0.37222222)]])
        self.stride = [4,8,16,32,64]
        self.clip = True

    def __call__(self,img_size):
        priors = []
        features_maps = []
        for stride in self.stride:
            features_maps.append((int(img_size[0]/stride),int(img_size[1]/stride)))  # 添加方形检测框
        for k , (feature_map_w, feature_map_h) in enumerate(features_maps):
            cwh = self.anchor_wh[k]
            for i in range(feature_map_w):
                for j in range(feature_map_h):
                    cx = (j + 0.5) / feature_map_w
                    cy = (i + 0.5) / feature_map_h
                    for m in range(len(cwh)):
                        priors.append([cx, cy, cwh[m][0], cwh[m][1]])
        priors = torch.tensor(priors)
        if self.clip:   # 对超出图像范围的框体进行截断
            priors = center_form_to_corner_form(priors) # 截断时,先转为 [xmin, ymin, xmin, xmax]形式
            priors.clamp_(max=1, min=0)
            priors = corner_form_to_center_form(priors) # 转回 [x, y, w, h]形式
        return priors


class Anchors_4:
    """
    Retainnet anchors, 生成策略与SSD不同
    """
    def __init__(self):

        if cfg.data_type == 'airport':
            self.anchor_wh = np.array([[(0.01666667, 0.06018519), (0.00347222, 0.01666667), (0.0109375,  0.01574074), (0.01458333, 0.03240741),(0.02916667, 0.0212963)],
                                       [(0.09479167, 0.2422619), (0.09114583, 0.08518519), (0.04270833, 0.03518519), (0.03194444, 0.06296296),(0.06302083, 0.05555556)],
                                      [(0.22552083, 0.225),(0.203125,   0.14074074), (0.15486111, 0.1462963),(0.12395833, 0.10185185),(0.07152778, 0.12314815)] ,
                                      [(0.35208333, 0.5787037), (0.39427083, 0.46481481), (0.40833333, 0.37222222),(0.37760417, 0.57037037),(0.38125,    0.3037037)]])

        elif cfg.data_type == 'M3FD':
            self.anchor_wh = np.array([[(0.01367188, 0.04427083), (0.01074219, 0.01432292), (0.01953125, 0.06901042),
                                        (0.02246094, 0.0234375), (0.00878906, 0.02734375)],
                                       [(0.04101562, 0.12239583), (0.07128906, 0.04427083), (0.02832031, 0.09375 ),
                                        (0.04785156, 0.05859375), (0.03320312, 0.03645833)],
                                       [(0.13476562, 0.14973958), (0.07910156 ,0.2734375), (0.109375,   0.09765625),
                                        (0.05273438, 0.1875), (0.07324219, 0.08203125)],
                                       [(0.59765625,0.48697917), (0.30664062, 0.28515625), (0.22167969, 0.18880208),
                                        (0.13710937, 0.38736979), (0.20410156, 0.12369792)]])

        self.stride = [4,8,16,32]
        self.clip = True

    def __call__(self,img_size):
        priors = []
        features_maps = []
        for stride in self.stride:
            features_maps.append((int(img_size[0]/stride),int(img_size[1]/stride)))  # 添加方形检测框
        for k , (feature_map_w, feature_map_h) in enumerate(features_maps):
            cwh = self.anchor_wh[k]
            for i in range(feature_map_w):
                for j in range(feature_map_h):
                    cx = (j + 0.5) / feature_map_w
                    cy = (i + 0.5) / feature_map_h
                    for m in range(len(cwh)):
                        priors.append([cx, cy, cwh[m][0], cwh[m][1]])
        priors = torch.tensor(priors)
        if self.clip:   # 对超出图像范围的框体进行截断
            priors = center_form_to_corner_form(priors) # 截断时,先转为 [xmin, ymin, xmin, xmax]形式
            priors.clamp_(max=1, min=0)
            priors = corner_form_to_center_form(priors) # 转回 [x, y, w, h]形式
        return priors

class Anchors:
    """
    Retainnet anchors, 生成策略与SSD不同
    """
    def __init__(self):
        # self.features_maps = [(56, 56 ), (28, 28), (14, 14)]

        # self.anchor_wh = np.array([[(0.00347222, 0.01666667), (0.01145833, 0.01666667), (0.03177083, 0.02592593)],
        #                              [(0.04930556, 0.05555556),(0.01770833, 0.05833333), (0.10885417, 0.09907407)] ,
        #                              [(0.18958333, 0.15092593), (0.35260417, 0.30595238), (0.396875, 0.46388889)]])\
        if cfg.data_type == 'voc':
            self.anchor_wh = np.array([[(0.034,0.064 ), (0.086,0.08 ), (0.054,0.15466667), (0.134,0.14578005),(0.088,0.28)],
                                      [(0.168,0.26933333),(0.156, 0.496 ), (0.156,0.496 ),(0.294,0.19393939),(0.276, 0.66966967)] ,
                                      [(0.416,0.47733333), (0.5, 0.7957958), (0.628,0.32266667),(0.788,0.58133333),(0.896,0.898   )]])

        elif cfg.data_type == 'airport':
            self.anchor_wh = np.array([[(0.00347222, 0.01666667), (0.01145833, 0.01574074), (0.015625 , 0.05277778), (0.028125, 0.0212963),(0.0296875,  0.06481481)],
                                      [(0.04270833, 0.03518519),(0.059375 , 0.05925926), (0.09114583, 0.08518519),(0.09479167, 0.1462963),(0.12447917, 0.10092593)] ,
                                      [(0.3859375,  0.31296296), (0.39322917, 0.46574074), (0.22552083, 0.225),(0.203125, 0.14074074),(0.15572917, 0.14537037)]])
        elif cfg.data_type == 'kitti':
            self.anchor_wh = np.array([[(0.0273752,0.216), (0.0410628,0.10666667), (0.08615137,0.2), (0.05716586,0.14666667),(0.01449275,0.11621622)],
                                      [(0.10708535,0.13333333),(0.1763285,0.40533333), (0.26731079,0.48266667),(0.14251208,0.24533333),(0.05877617,0.42933333)] ,
                                      [(0.04991948,0.06133333), (0.07246377,0.09333333), (0.02254428,0.05066667),(0.02979066,0.07466667),(0.01207729,0.03466667)]])
        # self.anchor_wh = np.array([[(0.00347222, 0.01666667),(0.01770833, 0.06666667), (0.01145833, 0.01574074), (0.01458333, 0.04166667),(0.04270833, 0.03518519),(0.02777778, 0.05648148),(0.02847222, 0.0212963),(0.04010417, 0.06309524)],
        #                           [(0.05885417, 0.06851852),(0.0791666, 0.05277778),(0.09114583, 0.08518519),(0.0947916, 0.1797619),(0.12291667, 0.09722222),(0.12708333, 0.12685185), (0.17881944, 0.225  ),(0.15694444, 0.15)] ,
        #                           [(0.19947917, 0.13888889), (0.18020833, 0.30595238), (0.28072917, 0.38703704),(0.22552083, 0.19814815),(0.3859375,  0.30740741),(0.390625,   0.46944444),(0.30052083, 0.26574074),(0.41666667, 0.37222222)]])
        self.stride = [8,16,32]
        self.image_size = 448
        self.clip = True

    def __call__(self,img_size):
        priors = []
        features_maps = []
        for stride in self.stride:
            features_maps.append((int(img_size[0]/stride),int(img_size[1]/stride)))  # 添加方形检测框
        for k , (feature_map_w, feature_map_h) in enumerate(features_maps):
            cwh = self.anchor_wh[k]
            for i in range(feature_map_w):
                for j in range(feature_map_h):
                    cx = (j + 0.5) / feature_map_w
                    cy = (i + 0.5) / feature_map_h
                    for m in range(len(cwh)):
                        priors.append([cx, cy, cwh[m][0], cwh[m][1]])
        priors = torch.tensor(priors)
        if self.clip:   # 对超出图像范围的框体进行截断
            priors = center_form_to_corner_form(priors) # 截断时,先转为 [xmin, ymin, xmin, xmax]形式
            priors.clamp_(max=1, min=0)
            priors = corner_form_to_center_form(priors) # 转回 [x, y, w, h]形式
        return priors



class Anchors_:
    """
    Retainnet anchors, 生成策略与SSD不同
    """
    def __init__(self,cfg=None):
        # self.features_maps = [(56, 56 ), (28, 28), (14, 14)]
        self.anchor_wh = np.array([(0.00347222, 0.01666667), (0.01145833, 0.01666667), (0.03177083, 0.02592593),
                                     (0.04930556, 0.05555556),(0.01770833, 0.05833333), (0.10885417, 0.09907407) ,
                                     (0.18958333, 0.15092593), (0.35260417, 0.30595238), (0.396875, 0.46388889)])
        self.stride = [8,16,32]
        self.image_size = 448
        self.clip = True

    def __call__(self,img_size):
        priors = []
        features_maps = []
        for stride in self.stride:
            features_maps.append((int(img_size[0]/stride),int(img_size[1]/stride)))  # 添加方形检测框
        cwh = self.anchor_wh
        for k , (feature_map_w, feature_map_h) in enumerate(features_maps):
            for i in range(feature_map_w):
                for j in range(feature_map_h):
                    cx = (j + 0.5) / feature_map_w
                    cy = (i + 0.5) / feature_map_h
                    for m in range(len(cwh)):
                        priors.append([cx, cy, cwh[m][0], cwh[m][1]])
        priors = torch.tensor(priors)
        if self.clip:   # 对超出图像范围的框体进行截断
            priors = center_form_to_corner_form(priors) # 截断时,先转为 [xmin, ymin, xmin, xmax]形式
            priors.clamp_(max=1, min=0)
            priors = corner_form_to_center_form(priors) # 转回 [x, y, w, h]形式
        return priors

class Anchors_new:
    """
    Retainnet anchors, 生成策略与SSD不同
    """
    def __init__(self,cfg=None):
        # self.features_maps = [(56, 56 ), (28, 28), (14, 14)]
        self.anchor_wh = np.array([[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]])
        self.stride = [8,16,32]
        self.ratios = np.array([0.5, 1, 2])
        self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        self.image_size = 448
        self.clip = True

    def __call__(self,img_size):
        priors = []
        features_maps = []
        for stride in self.stride:
            features_maps.append((int(img_size[0]/stride),int(img_size[1]/stride)))  # 添加方形检测框
        for k , (feature_map_w, feature_map_h) in enumerate(features_maps):
            cwh = self.anchor_wh[k] / feature_map_w
            for i in range(feature_map_w):
                for j in range(feature_map_h):
                    cx = (j + 0.5) / feature_map_w
                    cy = (i + 0.5) / feature_map_h
                    for m in range(len(cwh)):
                        priors.append([cx, cy, cwh[m][0], cwh[m][1]])
                    # priors.append([cx, cy, cwh])
                    # size = self.anchor_sizes[k]/self.image_size    # 将框体长宽转为 比例形式
                    #
                    # sides_square = self.scales * size   # 计算方形检测框边长
                    # for side_square in sides_square:
                    #     priors.append([cx, cy, side_square, side_square])   # 添加方形检测框
                    #
                    # sides_long = sides_square*(2**(1/2))  # 计算长形检测框长边
                    # for side_long in sides_long:
                    #     priors.append([cx, cy, side_long, side_long/2]) # 添加长形检测框,短边为长边的一半
                    #     priors.append([cx, cy, side_long/2, side_long])

        priors = torch.tensor(priors)
        if self.clip:   # 对超出图像范围的框体进行截断
            priors = center_form_to_corner_form(priors) # 截断时,先转为 [xmin, ymin, xmin, xmax]形式
            priors.clamp_(max=1, min=0)
            priors = corner_form_to_center_form(priors) # 转回 [x, y, w, h]形式
        return priors



class Anchors_1:
    """
    Retainnet anchors, 生成策略与SSD不同
    """
    def __init__(self,cfg=None):
        # self.features_maps = [(56, 56 ), (28, 28), (14, 14)]
        self.anchor_sizes = [8,32,64] #[8,16,32]才46.6  [16,32,64  58.3]
        self.stride = [8,16,32]
        self.ratios = np.array([0.5, 1, 2])
        self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        self.image_size = 448
        self.clip = True

    def __call__(self,img_size):
        priors = []
        features_maps = []
        for stride in self.stride:
            features_maps.append((int(img_size[0]/stride),int(img_size[1]/stride)))  # 添加方形检测框
        for k , (feature_map_w, feature_map_h) in enumerate(features_maps):
            for i in range(feature_map_w):
                for j in range(feature_map_h):
                    cx = (j + 0.5) / feature_map_w
                    cy = (i + 0.5) / feature_map_h

                    size = self.anchor_sizes[k]/self.image_size    # 将框体长宽转为 比例形式

                    sides_square = self.scales * size   # 计算方形检测框边长
                    for side_square in sides_square:
                        priors.append([cx, cy, side_square, side_square])   # 添加方形检测框

                    sides_long = sides_square*(2**(1/2))  # 计算长形检测框长边
                    for side_long in sides_long:
                        priors.append([cx, cy, side_long, side_long/2]) # 添加长形检测框,短边为长边的一半
                        priors.append([cx, cy, side_long/2, side_long])

        priors = torch.tensor(priors)
        if self.clip:   # 对超出图像范围的框体进行截断
            priors = center_form_to_corner_form(priors) # 截断时,先转为 [xmin, ymin, xmin, xmax]形式
            priors.clamp_(max=1, min=0)
            priors = corner_form_to_center_form(priors) # 转回 [x, y, w, h]形式
        return priors

if __name__ == '__main__':
    anchors = Anchors_1()([448,448])
    print(anchors[-10:])
    print(len(anchors))
