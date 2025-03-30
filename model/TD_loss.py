import sys
sys.path.append("../utils")
import torch
import torch.nn as nn
from utils import tools
import config.config_voc as cfg
import utils.anchors as anchors
import numpy as np
import utils.data_augment as dataAug
from utils.Boxs_op import center_form_to_corner_form, assign_priors,assign_priors_new,\
    corner_form_to_center_form, convert_boxes_to_locations, convert_locations_to_boxes
from utils.Focal_Loss import focal_loss
from utils import box_utils
import cv2

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels.cpu()]            # [N,D]

def LabelSmooth(onehot, num_classes, delta=0.01):
    return onehot * (1 - delta) + delta * 1.0 / num_classes

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)

        return loss

class TD_Loss(nn.Module):
    def __init__(self, iou_threshold_loss=0.5):
        super(TD_Loss, self).__init__()
        self.__iou_threshold_loss = iou_threshold_loss
        #self.anchor = anchors.Anchors_4()
        self.anchor = anchors.Anchors_4()
        # self.__cls_loss = focal_loss(alpha=0.25, gamma=2, num_classes=3, reduction='sum')

    def __creat_target(self,bboxes,center_form_anchors,imgs):

        bboxes_corner_form = bboxes[:,:4] #xyxy
        bbox_class_ind = bboxes[:,4:]

        conner_form_anchors = center_form_to_corner_form(center_form_anchors)
        #conner_form
        boxes, labels, conf = assign_priors_new(bboxes_corner_form, bbox_class_ind ,
                                     conner_form_anchors,imgs)
        # boxes, labels, conf = assign_priors(bboxes_corner_form, bbox_class_ind ,
        #                                conner_form_anchors)

        boxes = corner_form_to_center_form(boxes)

        return boxes,labels,conf

    def creat_label(self,img_size,bboxes,imgs):

        center_form_anchors = self.anchor(img_size).to(bboxes.device)

        all_bboxes_all = torch.zeros((0,center_form_anchors.shape[0],center_form_anchors.shape[1])).to(bboxes.device).float()
        all_labels_all = torch.zeros((0, center_form_anchors.shape[0],2)).to(bboxes.device).float()
        all_conf_all = torch.zeros((0, center_form_anchors.shape[0])).to(bboxes.device).float()
        all_bboxes_xywh = torch.zeros((0, bboxes.shape[1],4)).to(bboxes.device).float()

        for n in range(bboxes.shape[0]):
            bboxes_center_form = corner_form_to_center_form(bboxes[n,:,:4] )
            bboxes_annotation = bboxes[n, :, :]
            bboxes_annotation = bboxes_annotation[bboxes_annotation[:, 0] != -1,:]
            bboxes_all, labels_all, conf_all = self.__creat_target(bboxes_annotation, center_form_anchors,imgs)
            all_bboxes_all = torch.cat((all_bboxes_all, bboxes_all.unsqueeze(0)),0)
            all_labels_all = torch.cat((all_labels_all, labels_all.unsqueeze(0)), 0)
            all_conf_all = torch.cat((all_conf_all, conf_all.unsqueeze(0)), 0)
            all_bboxes_xywh = torch.cat((all_bboxes_xywh, bboxes_center_form.unsqueeze(0)), 0)

        del center_form_anchors
        return all_bboxes_all, all_labels_all, all_conf_all, all_bboxes_xywh


    def forward(self,batch_size, img_size, p, p_d, bboxes,imgs):

        bboxes_assigned, labels_assigned, conf_assigned, bboxes_xywh = self.creat_label(img_size, bboxes,imgs)

        CE = nn.CrossEntropyLoss(reduction="none")
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none")

        pred_d = p_d  # pred_d经过了坐标偏移到原始坐标的转换，center_form,clas也经过了sigmoid

        p_d_conf = p[..., 4]
        p_d_cls = p[..., 5:]
        p_d_xywh = pred_d[..., :4]

        label_xywh = bboxes_assigned[..., :4]  # batch*anchor*4
        # label_cls = label_assigned[...,0]
        class_num = p_d_cls.shape[-1]
        # label_cls = (label_assigned[...,0]-1) if (label_assigned[...,0]>0) else 0
        label_cls = labels_assigned[..., 0]#这里处理一下是因为后面有地方计算的时候，label不能为负值
        # label_cls = labels_assigned[..., 0] - 1
        # label_conf = int(label_assigned[..., 0] > 0)
        label_mix = labels_assigned[..., 1]

        label_obj_mask = (conf_assigned[...] == 1)  #正样本
        label_noobj_mask = (conf_assigned[...] == 0) #背景

        # iou = tools.iou_xywh_torch(p_d_xywh.unsqueeze(2), bboxes_xywh.unsqueeze(1))
        # iou_max = iou.max(-1, keepdim=True)[0]
        # label_noobj_mask = label_noobj_mask & (iou_max.squeeze(-1) < self.__iou_threshold_loss)

        # loss_conf = (label_obj_mask.float() + label_noobj_mask.float()) * FOCAL(input=p_d_conf,
        #                                                                         target=label_obj_mask.float()) * label_mix
        loss_conf = label_obj_mask.float()*FOCAL(input=p_d_conf,target=label_obj_mask.float())* label_mix \
                    + label_noobj_mask.float() * FOCAL(input=p_d_conf,target=label_obj_mask.float())

        #
        if type(p_d_xywh) is np.ndarray:
            p_d_xywh = torch.from_numpy(p_d_xywh).float()
        # loss giou
        # giou = tools.GIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)

        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size[0] ** 2)
        # loss_giou = label_obj_mask * bbox_loss_scale.squeeze(-1) * (1.0 - giou.squeeze(-1)) * label_mix

        predicted_boxes_xyxy = center_form_to_corner_form(p_d_xywh)
        gt_boxes_xyxy = center_form_to_corner_form(label_xywh)
        ciou = box_utils.bbox_overlaps_ciou(predicted_boxes_xyxy.view(-1, 4)[label_obj_mask.view(-1),:], gt_boxes_xyxy.view(-1, 4)[label_obj_mask.view(-1),:])
        # diou = box_utils.bbox_overlaps_diou(predicted_boxes_xyxy.view(-1, 4), gt_boxes_xyxy.view(-1, 4))
        # giou = box_utils.bbox_overlaps_giou(predicted_boxes_xyxy.view(-1, 4), gt_boxes_xyxy.view(-1, 4))

        loss_ciou = (bbox_loss_scale.view(-1)[label_obj_mask.view(-1)]) * (1.0 - ciou) * label_mix.view(-1)[label_obj_mask.view(-1)]

        # loss classes
        # 如果要用focal loss ,注意p_d_cls是已经经过sigmoid的
        # loss_cls = self.__cls_loss(p_d_cls[label_obj_mask|label_noobj_mask],label_cls[label_obj_mask|label_noobj_mask].long()) * (label_mix[label_obj_mask|label_noobj_mask].unsqueeze(-1))

        label_cls_one_hot = one_hot_embedding(label_cls.long(), p_d_cls.shape[-1]).to(p_d_cls.device)
        label_cls_one_hot = LabelSmooth(label_cls_one_hot, p_d_cls.shape[-1])
        loss_cls = BCE(input=p_d_cls.view(-1, p_d_cls.shape[-1])[label_obj_mask.view(-1)], target=label_cls_one_hot.view(-1, label_cls_one_hot.shape[-1])[label_obj_mask.view(-1)]) * label_mix.view(-1)[label_obj_mask.view(-1)].unsqueeze(-1)
        # loss_cls = label_obj_mask.int().reshape(-1) * CE(input=p_d_cls.reshape(-1, class_num),
        #                                       target=label_cls.reshape(-1).long()) * label_mix.reshape(-1)
        # loss_cls = label_obj_mask.int().unsqueeze(-1) * FOCAL(input=p_d_cls,
        #                                                       target=label_cls_one_hot) * label_mix.unsqueeze(-1)
        '''
        one_hot = self.one_hot_embedding(label_cls[label_obj_mask].long().cpu(), 4).to(device)  # [N,21]
        one_hot_smooth = dataAug.LabelSmooth()(one_hot, 4)

        loss_cls = BCE(input=p_cls[label_obj_mask], target=one_hot_smooth) * (label_mix[label_obj_mask].unsqueeze(-1))
        '''
        # loss_ciou = (torch.sum(loss_ciou)) / batch_size / torch.clamp(label_obj_mask.sum().float(), min=1.0)
        # loss_cls = (torch.sum(loss_cls)) / batch_size / torch.clamp(label_obj_mask.sum().float(), min=1.0)
        # loss_conf = (torch.sum(loss_conf)) / batch_size / torch.clamp(label_obj_mask.sum().float(), min=1.0)
        loss_ciou = (torch.sum(loss_ciou)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss = loss_ciou + loss_cls + loss_conf

        return loss, loss_ciou, loss_cls, loss_conf


if __name__ == "__main__":
    from model.TD_net import TD_Net
    net = TD_Net()

    p, p_d = net(torch.rand(3, 3, 416, 416))
    label_sbbox = torch.rand(3,  52, 52, 3,26)
    label_mbbox = torch.rand(3,  26, 26, 3, 26)
    label_lbbox = torch.rand(3, 13, 13, 3,26)
    sbboxes = torch.rand(3, 150, 4)
    mbboxes = torch.rand(3, 150, 4)
    lbboxes = torch.rand(3, 150, 4)

    loss, loss_xywh, loss_conf, loss_cls = YoloV3Loss(cfg.MODEL["ANCHORS"], cfg.MODEL["STRIDES"])(p, p_d, label_sbbox,
                                    label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
    print(loss)
