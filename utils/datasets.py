# coding=utf-8
import os
import sys
sys.path.append("..")
sys.path.append("../utils")
import torch
from torch.utils.data import Dataset, DataLoader
import config.yolov3_config_voc as cfg
import cv2
import numpy as np
import random
# from . import data_augment as dataAug
# from . import tools
import torchvision
import utils.data_augment as dataAug
import utils.tools as tools
import utils.anchors as anchors
from utils.Boxs_op import center_form_to_corner_form, assign_priors,\
    corner_form_to_center_form, convert_boxes_to_locations, convert_locations_to_boxes
from utils.visualize import *
from utils import Transfroms as transfrom

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]  #此处label的值不能为-1，否则直接变为倒数第一个class的index

def boxes_nms(boxes, scores, nms_thresh, max_count=-1):
    """ Performs non-maximum suppression, run on GPU or CPU according to
    boxes's device.
    Args:
        boxes(Tensor): `xyxy` mode boxes, use absolute coordinates(or relative coordinates), shape is (n, 4)
        scores(Tensor): scores, shape is (n, )
        nms_thresh(float): thresh
        max_count (int): if > 0, then only the top max_proposals are kept  after non-maximum suppression
    Returns:
        indices kept.
    """
    keep = torchvision.ops.nms(boxes, scores, nms_thresh)
    if max_count > 0:
        keep = keep[:max_count]
    return keep

class VocDataset(Dataset):
    def __init__(self, anno_file_type, img_size=416):
        self.img_size = img_size  # For Multi-training
        self.classes = cfg.DATA["CLASSES"]
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.__annotations = self.__load_annotations(anno_file_type)
        # self.__anchors = anchors.Anchors()
        self.corner_form_priors = 0.1

    def __len__(self):
        return  len(self.__annotations)

    def __getitem__(self, item):
        img_org, bboxes_org = self.__parse_annotation(self.__annotations[item])
        # import cv2 as cv
        # cv.imshow("img", img_org)
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW

        item_mix = random.randint(0, len(self.__annotations) - 1)
        img_mix, bboxes_mix = self.__parse_annotation(self.__annotations[item_mix])
        img_mix = img_mix.transpose(2, 0, 1)

        img, bboxes = dataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix

        # img_sample = transfrom.Resize(1080)(np.copy(img))
        img = torch.from_numpy(img).float()
        bboxes = torch.from_numpy(bboxes).float()
        bboxes_sample = torch.ones((100, 6))*(-1)
        bbox_count = 0
        for i in range(bboxes.shape[0]):
            bboxes_sample[int(bbox_count % 100), :] = bboxes[i, :]
            bbox_count += 1
        sample = {'img': img, 'bboxes': bboxes_sample}
        return sample

    def __getitem__1(self, item):
        # img_size = random.randint(5, 15) * 64
        # print("multi_scale_img_size : {}".format(img_size))
        img_size = 320

        img_org, bboxes_org = self.__parse_annotation(self.__annotations[item],img_size)
        # import cv2 as cv
        # cv.imshow("img", img_org)
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW

        item_mix = random.randint(0, len(self.__annotations) - 1)
        img_mix, bboxes_mix = self.__parse_annotation(self.__annotations[item_mix],img_size)
        img_mix = img_mix.transpose(2, 0, 1)

        img, bboxes = dataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix

        # img_sample = transfrom.Resize(1080)(np.copy(img))
        img = torch.from_numpy(img).float()
        bboxes = torch.from_numpy(bboxes).float()
        bboxes_sample = torch.ones((100, 6))*(-1)
        bbox_count = 0
        for i in range(bboxes.shape[0]):
            bboxes_sample[int(bbox_count % 100), :] = bboxes[i, :]
            bbox_count += 1
        sample = {'img': img, 'bboxes': bboxes_sample}
        return sample

        # img_p=visualize_boxes(image=img.transpose(1, 2, 0)*255, boxes=bboxes[:,:4], labels=bboxes[:,4].astype(np.int32), probs=bboxes[:,5], class_labels=self.classes)
        # path = os.path.join(cfg.PROJECT_PATH, "data/results/{}.jpg".format(1))
        # cv2.imwrite(path, img_p)

        center_form_anchors = self.__anchors(img.shape[1:3])

        # bboxes_location_all = np.zeros((0, 4)).astype(np.float32)
        # labels_all = np.zeros((0, 2)).astype(np.float32)
        # for i in range(len(center_form_anchors)):
        #     anchors = torch.from_numpy(center_form_anchors[i]).float()
        #     bboxes_location, labels = self.__creat_target(bboxes, anchors)
        #     bboxes_location_all = np.append(bboxes_location_all, bboxes_location, axis=0)
        #     labels_all = np.append(labels_all, labels, axis=0)

        # center_form_anchors_all = np.zeros((0, 4)).astype(np.float32)
        # for i in range(len(center_form_anchors)):
        #     center_form_anchors_all  = np.append(center_form_anchors_all, center_form_anchors[i], axis=0)
        bboxes_xywh = torch.zeros((150, 4))  # Darknet the max_num is 30
        bboxes_corner_form = bboxes[:, :4]  # xyxy
        bboxes_center_form = corner_form_to_center_form(bboxes_corner_form)
        bbox_count = 0
        for i in range(bboxes_center_form.shape[0]):
            bboxes_xywh[int(bbox_count % 150),:] = bboxes_center_form[i,:]
            bbox_count += 1
        bboxes_all, labels_all = self.__creat_target(bboxes, center_form_anchors)

        return sample
        return img, bboxes_all, labels_all, bboxes_xywh

        # center_form_anchors_all = np.zeros((0, 4)).astype(np.float32)
        # for i in range(len(center_form_anchors)):
        #     center_form_anchors_all  = np.append(center_form_anchors_all, center_form_anchors[i], axis=0)
        # center_form_anchors = torch.from_numpy(center_form_anchors).float()
        # bboxes_location_all = torch.from_numpy(bboxes_location_all).float()
        # labels_all = torch.from_numpy(labels_all).float()

        boxes = convert_locations_to_boxes(
            bboxes_location_all, center_form_anchors
        )
        mask = (labels_all[:, 0] >= 0)
        per_img_boxes = center_form_to_corner_form(boxes[mask])
        per_img_scores = one_hot_embedding(labels_all[:, 0][mask].long(), 4)
        processed_boxes = []
        processed_scores = []
        processed_labels = []
        for class_id in range(1, per_img_scores.size(1)):  # skip background
            scores = per_img_scores[:, class_id]
            mask = scores > 0.45
            scores = scores[mask]
            if scores.size(0) == 0:
                continue
            boxes = per_img_boxes[mask, :]
            boxes[:, 0::2] *= img.shape[1]
            boxes[:, 1::2] *= img.shape[2]

            keep = boxes_nms(boxes, scores, 0.45, 100)

            nmsed_boxes = boxes[keep, :]
            nmsed_labels = torch.tensor([class_id] * keep.size(0))
            nmsed_scores = scores[keep]

            processed_boxes.append(nmsed_boxes)
            processed_scores.append(nmsed_scores)
            processed_labels.append(nmsed_labels)

        if len(processed_boxes) == 0:
            processed_boxes = torch.empty(0, 4)
            processed_labels = torch.empty(0)
            processed_scores = torch.empty(0)
        else:
            processed_boxes = torch.cat(processed_boxes, 0)
            processed_labels = torch.cat(processed_labels, 0)
            processed_scores = torch.cat(processed_scores, 0)

        if processed_boxes.size(0) > self.cfg.MODEL.TEST.MAX_PER_IMAGE > 0:
            processed_scores, keep = torch.topk(processed_scores, k=self.cfg.MODEL.TEST.MAX_PER_IMAGE)
            processed_boxes = processed_boxes[keep, :]
            processed_labels = processed_labels[keep]


        return img, bboxes_location_all, labels_all

    def __creat_target(self,bboxes,center_form_anchors):

        bboxes_corner_form = bboxes[:,:4] #xyxy
        bbox_class_ind = bboxes[:,4:]

        conner_form_anchors = center_form_to_corner_form(center_form_anchors)
        #conner_form
        boxes, labels = assign_priors(bboxes_corner_form, bbox_class_ind ,
                                      conner_form_anchors)
        boxes = corner_form_to_center_form(boxes)

        return boxes,labels

    def __load_annotations(self, anno_type):

        assert anno_type in ['train', 'test'], "You must choice one of the 'train' or 'test' for anno_type parameter"
        if cfg.data_type == 'voc':
            anno_path = os.path.join(cfg.PROJECT_PATH, 'data','voc', anno_type+"_annotation.txt")
        elif cfg.data_type == 'airport':
            anno_path = os.path.join(cfg.PROJECT_PATH, 'data', 'airport', anno_type + "_annotation.txt")
        elif cfg.data_type == 'kitti':
            anno_path = os.path.join(cfg.PROJECT_PATH, 'data', 'kitti', anno_type + "_annotation.txt")
        with open(anno_path, 'r') as f:
            annotations = list(filter(lambda x:len(x)>0, f.readlines()))
        assert len(annotations)>0, "No images found in {}".format(anno_path)

        return annotations

    def __parse_annotation(self, annotation):
        """
        Data augument.
        :param annotation: Image' path and bboxes' coordinates, categories.
        ex. [image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...]
        :return: Return the enhanced image and bboxes. bbox'shape is [xmin, ymin, xmax, ymax, class_ind]
        """
        anno = annotation.strip().split(' ')

        img_path = anno[0]
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        # import cv2 as cv
        # cv.imshow("img", img)

        assert img is not None, 'File Not Found ' + img_path
        bboxes = np.array([list(map(float, box.split(','))) for box in anno[1:]])

        boxes = bboxes[:,:4]
        label = bboxes[:,4]
        # img, boxes, label = transfrom.ConvertFromInts()(np.copy(img), np.copy(boxes), np.copy(label))  # 图像数据转float32
        # img, boxes, label = transfrom.PhotometricDistort()(np.copy(img), np.copy(boxes), np.copy(label))  # 光度畸变,对比度,亮度,光噪声,色调,饱和等(详情看函数,有详细备注.)
        # img, boxes, label = transfrom.SubtractMeans([0, 0, 0] )(np.copy(img), np.copy(boxes), np.copy(label))  # 减均值
        # img, boxes, label = transfrom.DivideStds([1, 1, 1] )(np.copy(img), np.copy(boxes), np.copy(label))  # 除方差
        # img, boxes, label = transfrom.Expand()(np.copy(img), np.copy(boxes), np.copy(label))  # 随机扩充
        # img, boxes, label = transfrom.RandomSampleCrop()(np.copy(img), np.copy(boxes), np.copy(label))  # 随机交兵比裁剪
        # img, boxes, label = transfrom.RandomMirror()(np.copy(img), np.copy(boxes), np.copy(label))  # 随机镜像

        img, boxes = dataAug.RandomHorizontalFilp()(np.copy(img), np.copy(boxes))
        img, boxes = dataAug.RandomCrop()(np.copy(img), np.copy(boxes))
        img, boxes = dataAug.RandomAffine()(np.copy(img), np.copy(boxes))

        img, boxes, label = transfrom.ToPercentCoords()(np.copy(img), np.copy(boxes), np.copy(label))
        img, boxes, label = transfrom.Resize(self.img_size)(np.copy(img), np.copy(boxes), np.copy(label))

        # img, bboxes = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))
        return img, np.concatenate([boxes,label[:,np.newaxis]],axis=-1)

    def __creat_label(self, bboxes):
        """
        Label assignment. For a single picture all GT box bboxes are assigned anchor.
        1、Select a bbox in order, convert its coordinates("xyxy") to "xywh"; and scale bbox'
           xywh by the strides.
        2、Calculate the iou between the each detection layer'anchors and the bbox in turn, and select the largest
            anchor to predict the bbox.If the ious of all detection layers are smaller than 0.3, select the largest
            of all detection layers' anchors to predict the bbox.

        Note :
        1、The same GT may be assigned to multiple anchors. And the anchors may be on the same or different layer.
        2、The total number of bboxes may be more than it is, because the same GT may be assigned to multiple layers
        of detection.

        """

        anchors = np.array(cfg.MODEL["ANCHORS"])
        strides = np.array(cfg.MODEL["STRIDES"])
        train_output_size = self.img_size / strides
        anchors_per_scale = cfg.MODEL["ANCHORS_PER_SCLAE"]

        label = [np.zeros((int(train_output_size[i]), int(train_output_size[i]), anchors_per_scale, 6+self.num_classes))
                                                                      for i in range(3)]
        for i in range(3):
            label[i][..., 5] = 1.0

        bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]   # Darknet the max_num is 30
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_mix = bbox[5]

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = dataAug.LabelSmooth()(one_hot, self.num_classes)

            # convert "xyxy" to "xywh"
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # print("bbox_xywh: ", bbox_xywh)

            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((anchors_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # 0.5 for compensation
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = tools.iou_xywh_numpy(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:6] = bbox_mix
                    label[i][yind, xind, iou_mask, 6:] = one_hot_smooth

                    bbox_ind = int(bbox_count[i] % 150)  # BUG : 150为一个先验值,内存消耗大
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchors_per_scale)
                best_anchor = int(best_anchor_ind % anchors_per_scale)

                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:6] = bbox_mix
                label[best_detect][yind, xind, best_anchor, 6:] = one_hot_smooth

                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


if __name__ == "__main__":

    voc_dataset = VocDataset(anno_file_type="train", img_size=448)
    dataloader = DataLoader(voc_dataset, shuffle=True, batch_size=1, num_workers=0)

    for i, (img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(dataloader):
        if i==0:
            print(img.shape)
            print(label_sbbox.shape)
            print(label_mbbox.shape)
            print(label_lbbox.shape)
            print(sbboxes.shape)
            print(mbboxes.shape)
            print(lbboxes.shape)

            if img.shape[0] == 1:
                labels = np.concatenate([label_sbbox.reshape(-1, 26), label_mbbox.reshape(-1, 26),
                                         label_lbbox.reshape(-1, 26)], axis=0)
                labels_mask = labels[..., 4]>0
                labels = np.concatenate([labels[labels_mask][..., :4], np.argmax(labels[labels_mask][..., 6:],
                                        axis=-1).reshape(-1, 1)], axis=-1)

                print(labels.shape)
                tools.plot_box(labels, img, id=1)
