# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os
import config.config_voc as cfg


def prepare_data_path(data_dir_label,datatype):
    with open(data_dir_label, 'r') as f:
        lines = f.readlines()
        if datatype=='M3FD':
            filenames = [line.strip().split(' ')[0].split('\\')[-1] for line in lines]
            filenames_vi = [line.strip().split(' ')[0] for line in lines]
            filenames_ir = [line.strip().split(' ')[0].replace('vi','ir')for line in lines]
            label =  [line.strip().split(' ')[1:] for line in lines]
        if datatype=='airport':
            filenames = [line.strip().split(' ')[0].split('\\')[-1] for line in lines]
            filenames_vi = [line.strip().split(' ')[0].replace('JPEGImages','JPEGImages\\random_level\\airsport_smog_random')for line in lines]
            # filenames_vi = [line.strip().split(' ')[0] for line in lines]
            filenames_ir = [line.strip().split(' ')[0].replace('JPEGImages','IR').replace('.jpg','.png')for line in lines]
            label =  [line.strip().split(' ')[1:] for line in lines]
        if datatype=='FLIR':
            filenames = [line.strip().split(' ')[0].split('\\')[-1] for line in lines]
            filenames_vi = [line.strip().split(' ')[0] for line in lines]
            filenames_ir = [line.strip().split(' ')[0].replace('vi','ir')for line in lines]
            label =  [line.strip().split(' ')[1:] for line in lines]

    lenght = len(filenames_vi)
    return lenght,filenames,filenames_vi,filenames_ir,label

class Fusion_dataset(Dataset):
    def __init__(self, split,datatype,fusion_size=640, fusion_size_random=False,ir_path=None, vi_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test', 'fusion'], 'split must be "train"|"val"|"test"'

        self.size = fusion_size
        if fusion_size_random == True:
            self.hsize = torch.randint(600, 800, [1])
            self.wsize = torch.randint(600, 800, [1])
        else:
            self.hsize = self.size
            self.wsize = self.size

        if datatype == 'M3FD':
            if split == 'train':
                data_dir_label = './data/M3FD/train_annotation.txt'
            elif split == 'test':
                data_dir_label = './data/M3FD/test_annotation.txt'
            elif split == 'fusion':
                data_dir_label = './data/M3FD/train_annotation.txt'
        if datatype == 'airport':
            if split == 'train':
                data_dir_label = './data/airport/train_annotation.txt'
            elif split == 'test':
                data_dir_label = './data/airport/test_annotation.txt'
            elif split == 'night':
                data_dir_label = './data/airport/night_annotation.txt'
        if datatype == 'FLIR':
            if split == 'train':
                data_dir_label = './data/FLIR/train_annotation.txt'
            elif split == 'test':
                data_dir_label = './data/FLIR/test_annotation.txt'


        self.split = split
        self.resize = True
        self.img_size = cfg.TRAIN["TRAIN_IMG_SIZE"]
        self.len, self.filenames, self.filepath_vis, self.filepath_ir, self.label = prepare_data_path(data_dir_label,
                                                                                                      datatype)
        self.length = min(len(self.filepath_vis), len(self.filepath_ir))

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]
        filename = self.filenames[index]
        # label_path = self.filepath_label[index]
        # print(filename)
        # cv2.imread读取图片通道为BGR排列顺序
        image_vis = cv2.imread(vis_path, 1)
        image_ir = cv2.imread(ir_path, 0)
        height, width = image_ir.shape

        if self.resize==True:
            image_vis = cv2.resize(image_vis,(self.wsize , self.hsize))
            image_ir = cv2.resize(image_ir, (self.wsize, self.hsize))

        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.0
        image_ir = np.expand_dims(image_ir, axis=0) / 255.0

        image_vis = image_vis.astype(np.float32)
        image_ir = image_ir.astype(np.float32)

        try:
            bboxes = np.array([list(map(float, box.split(','))) for box in self.label[index]])[:,:4]
        except ValueError:
            bboxes = []
        lables = np.array([list(map(float, box.split(','))) for box in self.label[index]])[:,4]
        # box需要转换成比例
        # height, width, channels = image_ir.shape

        bboxes[:, 0] /= width
        bboxes[:, 2] /= width
        bboxes[:, 1] /= height
        bboxes[:, 3] /= height
        # bboxes_lables = np.concatenate([bboxes, lables[:, np.newaxis], np.full((len(bboxes), 1), 1.0)], axis=-1)
        bboxes_lables = np.concatenate([bboxes, lables[:, np.newaxis]], axis=-1)
        # np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)
        label_sample = np.ones((100, 5)) * (-1)
        bbox_count = 0
        for i in range(bboxes_lables.shape[0]):
            label_sample[int(bbox_count % 100), :] = bboxes_lables[i, :]
            bbox_count += 1

        return (
            torch.tensor(image_vis),
            torch.tensor(image_ir),
            torch.tensor(label_sample),
            filename
        )

    def __len__(self):
        return self.length

