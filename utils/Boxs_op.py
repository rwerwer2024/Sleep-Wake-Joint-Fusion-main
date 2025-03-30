import torch
# -*- coding: utf-8 -*-
# @Author  : LG
import math
# from tools import bbox_iou_torch_1
import config.config_voc as cfg

# 解码
def convert_locations_to_boxes(locations, priors):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    center_variance= 1
    size_variance = 1
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)


# 编码
def convert_boxes_to_locations(center_form_boxes, center_form_priors):
    # priors can have one dimension less
    center_variance= 1
    size_variance = 1
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)



def iou_of_1(boxes0, boxes1 , x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])

    union = (area0 + area1 - overlap_area + eps)
    inter = overlap_area
    iou = overlap_area / (area0 + area1 - overlap_area + eps)

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou
    return overlap_area / (area0 + area1 - overlap_area + eps)

def euclidean_dist(x, y):
    k, n = x.shape[0], y.shape[0]
    center_gt = torch.stack([(x[:,0]+x[:,2])/2.0,
                             (x[:,1]+x[:,3])/2.0], dim=1)
    center_bbox = torch.stack([(y[:,0]+y[:,2])/2.0,
                               (y[:,1]+y[:,3])/2.0], dim=1)

    # xx = torch.pow(center_gt, 2).sum(1, keepdim=True).expand(k, n)
    # yy = torch.pow(center_bbox, 2).sum(1, keepdim=True).expand(n, k).t()
    # dist = xx + yy
    # dist.addmm_(1, -2, center_gt, center_bbox.t())
    # dist = dist.sqrt()

    distances = (center_bbox[:, None, :] -
                 center_gt[None, :, :]).pow(2).sum(-1).sqrt().t()

    return distances, center_gt, center_bbox

def assign_priors_new(gt_boxes, gt_labels, corner_form_priors,imgs):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets

    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))

    center_dst_matrix, center_gt, center_bbox = euclidean_dist(gt_boxes, corner_form_priors)

    num_gts, num_bboxes = ious.size(1), ious.size(0)

    INF = 100000000
    overlaps_inf = torch.full_like(ious,-INF)

    for ii in range(num_gts):
        ious_dis = ious[:,ii]

        #1，先取iou的前30个
        iou_value, iou_top100_candidate_indexes = ious_dis.topk(10)

        #2，再再iou前30个里面去距离最近的10个作为候选
        center_dis = torch.ones(num_bboxes)*2
        center_dis = center_dis.to(gt_boxes.device)
        center_dis[iou_top100_candidate_indexes] = center_dst_matrix[ii][iou_top100_candidate_indexes].float()
        value, candidate_indexes = center_dis.topk(5, largest=False)

        #3，又再候选里面选择中心点落在了目标框内的样本
        bool_inds = (center_bbox[candidate_indexes][:, 0] > gt_boxes[ii][0]) & \
                    (center_bbox[candidate_indexes][:, 0] < gt_boxes[ii][2]) & \
                    (center_bbox[candidate_indexes][:, 1] > gt_boxes[ii][1]) & \
                    (center_bbox[candidate_indexes][:, 1] < gt_boxes[ii][3])

        pos_inds = candidate_indexes[bool_inds]
        # pos_inds = candidate_indexes
        overlaps_inf[:,ii][pos_inds] = ious[:,ii][pos_inds]

    #4，假设有一个anchor匹配了多个gt,选择iou最大的那个gt来匹配
    max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
    labels = torch.zeros([num_bboxes,2]).to(gt_boxes.device)
    boxes = torch.zeros([num_bboxes,4]).to(gt_boxes.device)
    conf = torch.zeros(num_bboxes).to(gt_boxes.device)
    labels[max_overlaps != -INF,:]=gt_labels[argmax_overlaps[max_overlaps != -INF],:].float()
    conf[max_overlaps != -INF] = 1
    boxes[max_overlaps != -INF,:] = gt_boxes[argmax_overlaps[max_overlaps != -INF]].float()

    #5,与gt的iou最大的那个anchor必须用来预测该gt,必须保证每一个样本最少有一个anchor负责预测
    best_prior_per_target, best_prior_per_target_index = ious.max(0)
    labels[best_prior_per_target_index] = gt_labels.float()
    conf[best_prior_per_target_index] = 1
    boxes[best_prior_per_target_index] = gt_boxes.float()
    '''
    boxes_show = boxes[conf==1].cpu().numpy()
    labels_show = labels[conf==1].cpu().numpy()
    import numpy as np
    import cv2 as cv
    ptLeftTop = (60, 60)
    img = (imgs[0].cpu().numpy().transpose(1,2,0)).astype('uint8').copy()
    for i in range(len(gt_boxes)):
        cv.rectangle(img, (int(gt_boxes[i, 0] * 512), int(gt_boxes[i, 1] * 512)),
                     (int(gt_boxes[i, 2] * 512), int(gt_boxes[i, 3] * 512)), (0, 255, 0))
    cv.imshow("img", img)
    cv.waitKey()
    cv.destroyAllWindows()

    # i = 0
    # for i in range((conf==1).sum()):# 生成一个空灰度图像
    #     cv.rectangle(img, (int(boxes_show[i,0]*512),int(boxes_show[i,1]*512)),(int(boxes_show[i,2]*512),int(boxes_show[i,3]*512)), (0, 0, 255))

    i = 0
    corner_form_priors_index = corner_form_priors[conf == 1]
    for i in range((conf == 1).sum()):  # 生成一个空灰度图像
        cv.rectangle(img, (int(corner_form_priors_index[i, 0] * 512), int(corner_form_priors_index[i, 1] * 512)),
                     (int(corner_form_priors_index[i, 2] * 512), int(corner_form_priors_index[i, 3] * 512)), (0, 0, 255))


    cv.imshow("img", img);
    '''
    '''
    #统计每个样本最终匹配的anchor数目，做这个实验的时候，mixup要取消
    a=torch.ones(num_gts)
    for i in range(num_gts):
        a[i]=i
    argmax_overlaps[best_prior_per_target_index] = a.long().cuda()
    b = torch.zeros(num_bboxes).cuda()
    b[best_prior_per_target_index]=1
    positive_index = (max_overlaps != -INF)|(b==1)

    anchor_for_gt =  torch.zeros(num_gts)
    positive_sample = argmax_overlaps[positive_index]
    # positive_sample = argmax_overlaps[max_overlaps != -INF]

    for k in range(num_gts):
        for j in range(len(positive_sample)):
            if (positive_sample[j] == k) :
                anchor_for_gt[k] += 1
        #写在文件里，方便统计
        if anchor_for_gt[k]==0:
            print('anchor_for_gt[k]==0')
        sample_num = '%d' % (anchor_for_gt[k])
        sample_class = int(gt_labels[k,0])
        s = ' '.join([sample_num]) + '\n'
        import os
        with open(os.path.join(cfg.PROJECT_PATH, 'data', 'sample_statistic', 'sample_statistics_' + cfg.DATA["CLASSES"][sample_class] + '.txt'), 'a') as f:
            f.write(s)

    '''
    return boxes, labels, conf



def assign_priors(gt_boxes, gt_labels, corner_form_priors):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets
    pos_iou_max_threshold = 0.4
    neg_iou_min_threshold = 0.4

    # ious1 = calc_iou(corner_form_priors.unsqueeze(1),gt_boxes.unsqueeze(0))
    # ious = bbox_iou_torch_1(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1)).squeeze(-1)
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index  #bug,有可能多个target对应的最匹配的anchor是同一个，这里只是默认这个anchor负责预测最后一个
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)  #保证best_prior_per_target_index不会因为ious小而被认为是负样本
    # size: num_priors
    #gt_labels包含了label 和 mix
    labels = gt_labels[best_target_per_prior_index]
    #置信度
    conf = torch.ones(labels.shape[0]).to(labels.device)  #正样本 1
    conf[best_target_per_prior < pos_iou_max_threshold] = -1  # 忽略 -1
    conf[best_target_per_prior < neg_iou_min_threshold] = 0  # the backgournd id 0
    #对类别的标签进行操作
    # labels[:,0] += 1 #类别数目+1,0为背景，-1为忽略
    # labels[best_target_per_prior < pos_iou_max_threshold,0] = -1  # 忽略
    # labels[best_target_per_prior < neg_iou_min_threshold,0] = 0  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels, conf


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask

# [x, y, w, h] to [xmin, ymin, xmax, ymax]
def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2,
                      locations[..., :2] + locations[..., 2:] / 2], locations.dim() - 1)

# [xmin, ymin, xmax, ymax] to [x, y, w, h]
def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)



if __name__ == '__main__':
    import os
    import numpy as np

    import numpy as np
    import matplotlib.pyplot as plt

    '''
    plt.subplot(1, 1, 1)
    fig = plt.figure()
    plt.figure(figsize=(8, 6))
    x1 = np.array([8556, 5335, 7310, 6482])
    x2 = np.array([4283, 2667, 3655, 3241])
    labels = ["东区", "北区", "南区", "西区"]
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    plt.pie(x1, labels=labels, radius=1.0,
            wedgeprops=dict(width=0.3, edgecolor='w'))
    plt.pie(x2, radius=0.7, wedgeprops=dict(width=0.3, edgecolor='w'))
    plt.annotate("完成量",
                 xy=(0.35, 0.35), xytext=(0.7, 0.45),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.annotate("任务量",
                 xy=(0.75, 0.20), xytext=(1.1, 0.2),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.title("全国各分区任务量和完成量", loc="center")
    plt.show()
    '''

    sample = {}
    anchor_sample = np.zeros([7,cfg.DATA["NUM"]],dtype=np.int)
    for i in range(cfg.DATA["NUM"]):
        with open(os.path.join(cfg.PROJECT_PATH, 'data', 'sample_statistic',
                               'sample_statistics_' + cfg.DATA["CLASSES"][i] + '.txt'), 'r') as f:
            lines = f.readlines()
        anchor_positive_num = [int(x.strip()) for x in lines]
        anchor_positive_num_numpy = np.array(anchor_positive_num)

        sample[cfg.DATA["CLASSES"][i]] = anchor_positive_num_numpy

        for j in range(len(anchor_positive_num)):
            if anchor_positive_num[j]==0:
                anchor_sample[0,i] += 1
            if anchor_positive_num[j]==1:
                anchor_sample[1,i] += 1
            if anchor_positive_num[j]==2:
                anchor_sample[2,i] += 1
            if anchor_positive_num[j]==3:
                anchor_sample[3,i] += 1
            if anchor_positive_num[j]==4:
                anchor_sample[4,i] += 1
            elif anchor_positive_num[j]==5:
                anchor_sample[5,i] += 1
            elif anchor_positive_num[j]==6:
                anchor_sample[6,i] += 1
            else:
                print('wrong anchor sample num!')


    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = ['0','1', '2', '3', '4', '5', '6']
    # A1 = [40, 25, 20, 5, 10]
    # A2 = [20, 35, 15, 18, 12]
    # A3 = [10, 23, 67, 3, 10]
    color = ['r','b','yellow','purple','lightgreen','lightblue','g']

    textprops = {'fontsize': 50, 'color': 'black'}

    # wedges2, texts2, autotexts2 = plt.pie(anchor_sample[:,2],
    #                                       autopct='%2.1f%%',
    #                                       radius=0.7,
    #                                       pctdistance=0.8,
    #                                       colors=color,
    #                                       startangle=180,
    #                                       textprops=textprops,
    #                                       wedgeprops=dict(width=0.3, edgecolor='w'))
    plt.pie(anchor_sample[:,1],

                                          radius=1,

                                          colors=color,
                                          startangle=180,
                                          wedgeprops=dict(width=0.3, edgecolor='w'))

    wedges2, texts2 = plt.pie(anchor_sample[:,2],

                                          radius=0.7,

                                          colors=color,
                                          startangle=180,

                                          wedgeprops=dict(width=0.3, edgecolor='w'))

    plt.pie(anchor_sample[:,0],

                                          radius=0.4,

                                          colors=color,
                                          startangle=180,
                                          textprops=textprops,
                                          wedgeprops=dict(width=0.3, edgecolor='w'))

    plt.legend(wedges2,
               labels,
               fontsize=12,
               title='anchor number',
               loc='center right',
               bbox_to_anchor=(0.91, 0, 0.3, 1))
    # plt.setp(autotexts1, size=15, weight='bold')
    # plt.setp(autotexts2, size=15, weight='bold')
    # plt.setp(autotexts3, size=15, weight='bold')
    # plt.setp(texts1, size=12)
    plt.title('anchor matching distribution')
    plt.show()



    '''  
    import numpy as np
    import matplotlib.pyplot as plt

    people = ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8')
    segments = 4

    # multi-dimensional data
    # data = np.asarray([[3.40022085, 7.70632498, 6.4097905, 10.51648577, 7.5330039,
    #                     7.1123587, 12.77792868, 3.44773477],
    #                    [11.24811149, 5.03778215, 6.65808464, 12.32220677, 7.45964195,
    #                     6.79685302, 7.24578743, 3.69371847],
    #                    [3.94253354, 4.74763549, 11.73529246, 4.6465543, 12.9952182,
    #                     4.63832778, 11.16849999, 8.56883433],
    #                    [4.24409799, 12.71746612, 11.3772169, 9.00514257, 10.47084185,
    #                     10.97567589, 3.98287652, 8.80552122]])
    # data = anchor_sample.transpose(1,0)
    data = anchor_sample
    percentages = np.zeros((20, 3))
    col_sum = np.sum(data, axis=0)
    for i in range(data.shape[0]):
        for j in range(len(data[i])):
            percentages[j, i] = data[i, j] / col_sum[j] * 100

    y_pos = np.arange(20)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    colors = 'rgm'
    patch_handles = []

    bottom = np.zeros(20)
    for i, d in enumerate(data):
        patch_handles.append(ax.bar(y_pos, d,
                                    color=colors[i % len(colors)], align='center',
                                    bottom=bottom))
        bottom += d

    # search all of the bar segments and annotate
    for j in range(len(patch_handles)):
        for i, patch in enumerate(patch_handles[j].get_children()):
            bl = patch.get_xy()
            x = 0.5 * patch.get_width() + bl[0]
            y = 0.5 * patch.get_height() + bl[1]
            ax.text(x, y, "%.2f%%" % (percentages[i, j]), ha='center',fontsize = 8)

    plt.show()
    '''
