#!/usr/bin/python
# -*- encoding: utf-8 -*-


from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
from loss import  Fusionloss
from optimizer import Optimizer
import torch
import utils.data_augment as dataAug
from eval.evaluator import *
from model.TD_net import TD_Net
from model.TD_loss import TD_Loss
from MyFusionNet import MyFusionNet3
from torch.utils.data import DataLoader
import config.config_voc as cfg
from utils import cosine_lr_scheduler
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def run_fusion(type='test'):
    fusion_model_path = './weight/end2end_last_fusion_epoch_stage1.pt'
    fused_dir = './Fusion_results/'

    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    fusionmodel = eval('MyFusionNet3')(output=1)
    fusionmodel.eval()
    if args.gpu >= 0:
        fusionmodel.cuda(args.gpu)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))

    test_dataset = Fusion_dataset(type,datatype=cfg.data_type)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, images_label, name) in enumerate(test_loader):

            image_vis = Variable(images_vis).cuda()
            # bboxes = Variable(images_label).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(images_ir).cuda()

            logits = fusionmodel(image_vis_ycrcb, image_ir)
            fusion_ycrcb = torch.cat(
                (logits, image_vis_ycrcb[:, 1:2, :, :],
                 image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)

            fusion_image = (fusion_image - torch.min(fusion_image)) / (
                    torch.max(fusion_image) - torch.min(fusion_image)
            )
            fusion_image = torch.round(255.0 * fusion_image).float()

            ###############################################
            for k in range(len(image_vis)):
                image = np.uint8(fusion_image[k, :, :, :].cpu().numpy())
                image = image.squeeze()
                image = image.transpose((1, 2, 0))
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))

def train_end2end_fusion( logger=None):
    batch_size = 2
    total_epoch = 10

    if logger == None:
        logger = logging.getLogger()

    end2end_fusion_model_name = './weight/end2end_last_fusion_epoch_stage1.pt'
    fusionmodel = eval('MyFusionNet3')(output=1)
    fusionmodel.cuda()
    fusionmodel.train()
    criteria_fusion = Fusionloss()

    train_dataset = Fusion_dataset('train',datatype=cfg.data_type)
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    optimizer1 = torch.optim.Adam(fusionmodel.parameters(), lr=1e-2)
    train_loader.n_iter = len(train_loader)
    st = glob_st = time.time()

    detect_loss, loss_giou, loss_cls, loss_conf = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)

    for epo in range(0, total_epoch):
        print('\n| epo #%s begin...' % epo)
        lr_start = 0.001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** epo
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr_this_epo

        for it, (image_vis, image_ir, image_label, image_name) in enumerate(train_loader):

            image_vis = Variable(image_vis).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()

            logits = fusionmodel(image_vis_ycrcb, image_ir)

            loss_fusion, loss_in, loss_grad ,loss_ssim = criteria_fusion(
                image_vis_ycrcb, image_ir, logits, 0
            )

            loss_fusion.backward()
            # torch.nn.utils.clip_grad_norm_(fusionmodel.parameters(), 1, norm_type=2)
            optimizer1.step()
            optimizer1.zero_grad()


            if ((it + 1) % 10) == 0:
                ed = time.time()

                t_intv, glob_t_intv = ed - st, ed - glob_st
                now_it = train_loader.n_iter * epo + it + 1
                eta = int((train_loader.n_iter * total_epoch - now_it)
                          * (glob_t_intv / (now_it)))
                eta = str(datetime.timedelta(seconds=eta))
                loss_detect = detect_loss.item()
                loss_fusion = loss_fusion.item()


                st = ed

                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'fusion_lr: {fusion_lr:.5f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_ssim: {loss_ssim:.4f}',
                        'loss_fusion: {loss_fusion:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * total_epoch,
                    fusion_lr=optimizer1.param_groups[0]['lr'],
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_ssim=loss_ssim.item(),
                    loss_fusion=loss_fusion,
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)


        fusion_model_file = os.path.join('weight', 'fusion_model_backup_epoch%g_stage1.pt' % epo)
        torch.save(fusionmodel.state_dict(), fusion_model_file)

    torch.save(fusionmodel.state_dict(), end2end_fusion_model_name)
    logger.info("Fusion Model Save to: {}".format(end2end_fusion_model_name))
    logger.info('\n')

def train_end2end_fusion_detect(logger=None):

    batch_size = 4
    total_epoch = 20
    best_mAP = 0

    if logger == None:
        logger = logging.getLogger()

    fusion_model_file = './weight/end2end_last_fusion_epoch_stage1.pt'
    detect_model_file = './weight/detect_model_backup_epoch10_stage2.pt'
    end2end_fusion_model_name = './weight/end2end_last_fusion_epoch_stage2.pt'
    end2end_detect_model_name = './weight/end2end_last_detect_epoch_stage2.pt'

    fusionmodel = eval('MyFusionNet3')(output=1)
    fusionmodel.cuda()
    fusionmodel.eval()
    criteria_fusion = Fusionloss()
    fusionmodel.load_state_dict(torch.load(fusion_model_file))
    logger.info('Load Pre-trained Fusion Model')
    for p in fusionmodel.parameters():
        p.requires_grad = False

    detect_model = TD_Net()
    # detect_model.load_state_dict(torch.load(detect_model_file))
    detect_model.cuda()
    detect_model.train()
    criterion = TD_Loss(iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])


    train_dataset = Fusion_dataset('train',datatype=cfg.data_type)
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    optimizer2 = torch.optim.SGD(detect_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
    scheduler = cosine_lr_scheduler.CosineDecayLR(optimizer2, T_max=total_epoch * len(train_loader),
                                                  lr_init=1e-2,
                                                  lr_min=1e-4,
                                                  warmup=5 * len(train_loader))

    train_loader.n_iter = len(train_loader)
    st = glob_st = time.time()
    logger.info('Training detection Model start~')

    for epo in range(0, total_epoch):
        print('\n| epo #%s begin...' % epo)

        for it, (image_vis, image_ir, image_label, image_name) in enumerate(train_loader):

            iteration = epo * len(train_loader) + it + 1
            scheduler.step(iteration)

            image_vis = Variable(image_vis).cuda()
            bboxes = Variable(image_label).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()

            logits = fusionmodel(image_vis_ycrcb, image_ir)
            fusion_ycrcb = torch.cat(
                (logits, image_vis_ycrcb[:, 1:2, :, :],
                 image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)

            # loss_fusion, loss_in, loss_grad ,loss_ssim = criteria_fusion(
            #     image_vis_ycrcb, image_ir, logits, 0
            # )

            fusion_image = (fusion_image - torch.min(fusion_image)) / (
                    torch.max(fusion_image) - torch.min(fusion_image)
            )
            fusion_image = torch.round(255.0 * fusion_image).float()

            fusion_image_BGR = fusion_image[:, [2, 1, 0], :, :]
            batch_size = fusion_image_BGR.shape[0]
            fusion_image_BGR = F.interpolate(fusion_image_BGR, cfg.TRAIN["TRAIN_IMG_SIZE"], mode='bilinear', align_corners=False)
            img_size = fusion_image_BGR.shape[2:]

            fusion_image_BGR_new= fusion_image_BGR.clone()
            bboxes_new = torch.ones((batch_size, 200, 6)).double().cuda() * (-1)

            #mixup数据增强
            ##########################################
            for i in range(0,batch_size):
                item_org= i
                item_mix = random.randint(0,batch_size-1)
                img_org = fusion_image_BGR[item_org,...]
                bboxes_org = bboxes[item_org,...]
                img_mix = fusion_image_BGR[item_mix,...]
                bboxes_mix = bboxes[item_mix,...]
                img_mixup, bboxes_mixup = dataAug.Mixup_torch()(img_org, bboxes_org, img_mix, bboxes_mix)
                fusion_image_BGR_new[item_org,...] = img_mixup
                for j in range(bboxes_mixup.shape[0]):
                    bboxes_new[item_org, j, :] = bboxes_mixup[j, :]
            ##########################################

            p, p_d = detect_model(fusion_image_BGR_new)
            detect_loss, loss_giou, loss_cls, loss_conf = criterion(batch_size, img_size, p, p_d, bboxes_new, fusion_image_BGR_new)

            detect_loss.backward()
            torch.nn.utils.clip_grad_norm_(detect_model.parameters(), 1, norm_type=2)

            if ((it + 1) % 1) == 0:
                # Accumulates gradient before each step
                optimizer2.step()  # 更新参数
                optimizer2.zero_grad()

            if ((it + 1) % 10) == 0:
                ed = time.time()

                t_intv, glob_t_intv = ed - st, ed - glob_st
                now_it = train_loader.n_iter * epo + it + 1
                eta = int((train_loader.n_iter * total_epoch - now_it)
                          * (glob_t_intv / (now_it)))
                eta = str(datetime.timedelta(seconds=eta))
                loss_detect = detect_loss.item()

                st = ed

                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'detect_lr: {detect_lr:.4f}',
                        'loss_giou: {loss_giou:.4f}',
                        'loss_cls: {loss_cls:.4f}',
                        'loss_conf: {loss_conf:.4f}',
                        'loss_detect: {loss_detect:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * total_epoch,
                    detect_lr=optimizer2.param_groups[0]['lr'],
                    loss_giou=loss_giou.item(),
                    loss_cls=loss_cls.item(),
                    loss_conf=loss_conf.item(),
                    loss_detect=loss_detect,
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)

        detect_model_file = os.path.join('weight', 'detect_model_backup_epoch%g_stage2.pt' % epo)
        torch.save(detect_model.state_dict(), detect_model_file)

        mAP = 0
        if epo >= 10:
            print('*' * 20 + "Validate" + '*' * 20)
            with torch.no_grad():
                APs = My_Evaluator(fusionmodel,detect_model).APs_voc()
                for i in APs:
                    print("{} --> mAP : {}".format(i, APs[i]))
                    mAP += APs[i]
                mAP = mAP / cfg.DATA["NUM"]
                print('mAP:%g' % (mAP))

            if mAP > best_mAP:
                best_mAP = mAP

            if  cfg.data_type == 'M3FD':
                best_weight = os.path.join('weight', "best_stage2_%g_%g_%g_%g_%g_%g_%g_%g.pt " % (
                    epo, APs['People'], APs['Car'], APs['Bus'], APs['Motorcycle'], APs['Truck'], APs['Lamp'], mAP))
            elif cfg.data_type == 'airport':
                best_weight = os.path.join('weight', "best_stage2_%g_%g_%g_%g_%g.pt " % (epo, APs['airplane'], APs['car'], APs['man'], mAP))

            if best_mAP == mAP:
                torch.save(detect_model.state_dict(), best_weight)
            print('best mAP : %g' % (best_mAP))

    torch.save(detect_model.state_dict(), end2end_detect_model_name)
    logger.info("Detect Model Save to: {}".format(end2end_detect_model_name))
    logger.info('\n')


def train_end2end( logger=None):
    batch_size = 4
    total_epoch = 20
    best_mAP = 0

    if logger == None:
        logger = logging.getLogger()

    detect_model_path = './weight/end2end_last_detect_epoch_stage2.pt'
    fusion_model_file = './weight/end2end_last_fusion_epoch_stage1.pt'

    end2end_fusion_model_name = './weight/end2end_last_fusion_epoch_stage3.pt'
    end2end_detect_model_name = './weight/end2end_last_detect_epoch_stage3.pt'

    fusionmodel = eval('MyFusionNet3')(output=1)
    fusionmodel.cuda()
    fusionmodel.train()
    criteria_fusion = Fusionloss()
    fusionmodel.load_state_dict(torch.load(fusion_model_file))
    logger.info('Load Pre-trained Fusion Model')

    detect_model = TD_Net()
    detect_model.cuda()
    detect_model.train()
    criterion = TD_Loss(iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

    detect_model.load_state_dict(torch.load(detect_model_path))
    logger.info('Load Pre-trained Detectin Model')

    train_dataset = Fusion_dataset('train',datatype=cfg.data_type)
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    optimizer1 = torch.optim.Adam(fusionmodel.parameters(), lr=1e-3)
    optimizer2 = torch.optim.SGD(detect_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)

    train_loader.n_iter = len(train_loader)
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')

    for epo in range(0, total_epoch):
        print('\n| epo #%s begin...' % epo)
        lr_start = 0.0001
        lr_decay = 0.9
        lr_this_epo = lr_start * lr_decay ** epo
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr_this_epo
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr_this_epo

        for it, (image_vis, image_ir, image_label, image_name) in enumerate(train_loader):

            image_vis = Variable(image_vis).cuda()
            bboxes = Variable(image_label).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()

            logits = fusionmodel(image_vis_ycrcb, image_ir)
            fusion_ycrcb = torch.cat(
                (logits, image_vis_ycrcb[:, 1:2, :, :],
                 image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)

            loss_fusion, loss_in, loss_grad ,loss_ssim = criteria_fusion(
                image_vis_ycrcb, image_ir, logits, 0
            )

            fusion_image = (fusion_image - torch.min(fusion_image)) / (
                    torch.max(fusion_image) - torch.min(fusion_image)
            )
            fusion_image = torch.round(255.0 * fusion_image).float()

            fusion_image_BGR = fusion_image[:, [2, 1, 0], :, :]
            batch_size = fusion_image_BGR.shape[0]
            fusion_image_BGR = F.interpolate(fusion_image_BGR, cfg.TRAIN["TRAIN_IMG_SIZE"], mode='bilinear', align_corners=False)
            img_size = fusion_image_BGR.shape[2:]

            fusion_image_BGR_new= fusion_image_BGR.clone()
            bboxes_new = torch.ones((batch_size, 200, 6)).double().cuda() * (-1)

            #mixup数据增强
            ##########################################
            for i in range(0,batch_size):
                item_org= i
                item_mix = random.randint(0,batch_size-1)
                img_org = fusion_image_BGR[item_org,...]
                bboxes_org = bboxes[item_org,...]
                img_mix = fusion_image_BGR[item_mix,...]
                bboxes_mix = bboxes[item_mix,...]
                img_mixup, bboxes_mixup = dataAug.Mixup_torch()(img_org, bboxes_org, img_mix, bboxes_mix)
                fusion_image_BGR_new[item_org,...] = img_mixup
                for j in range(bboxes_mixup.shape[0]):
                    bboxes_new[item_org, j, :] = bboxes_mixup[j, :]
            ##########################################

            p, p_d = detect_model(fusion_image_BGR_new)
            detect_loss, loss_giou, loss_cls, loss_conf = criterion(batch_size, img_size, p, p_d, bboxes_new, fusion_image_BGR_new)

            # 1,更新融合网络
            lamda = 2/total_epoch*(epo+1)
            loss_total = loss_fusion + lamda * detect_loss  # 检测损失权重不断增大
            loss_total.backward(retain_graph=True, inputs=list(fusionmodel.parameters()))
            # torch.nn.utils.clip_grad_norm_(fusionmodel.parameters(), 1, norm_type=2)

            # 2,更新检测网络
            # optimizer2.zero_grad()
            detect_loss.backward(inputs=list(detect_model.parameters()))
            # torch.nn.utils.clip_grad_norm_(detect_model.parameters(), 1, norm_type=2)

            if ((it + 1) % 2) == 0:
                torch.nn.utils.clip_grad_norm_(detect_model.parameters(), 2, norm_type=2)
                optimizer1.step()
                optimizer2.step()  # 更新参数
                optimizer1.zero_grad()
                optimizer2.zero_grad()

            if ((it + 1) % 10) == 0:
                ed = time.time()

                t_intv, glob_t_intv = ed - st, ed - glob_st
                now_it = train_loader.n_iter * epo + it + 1
                eta = int((train_loader.n_iter * total_epoch - now_it)
                          * (glob_t_intv / (now_it)))
                eta = str(datetime.timedelta(seconds=eta))
                loss_detect = detect_loss.item()
                loss_fusion = loss_fusion.item()


                st = ed

                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'fusion_lr: {fusion_lr:.4f}',
                        'detect_lr: {detect_lr:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_ssim: {loss_ssim:.4f}',
                        'loss_fusion: {loss_fusion:.4f}',
                        'loss_giou: {loss_giou:.4f}',
                        'loss_cls: {loss_cls:.4f}',
                        'loss_conf: {loss_conf:.4f}',
                        'loss_detect: {loss_detect:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * total_epoch,
                    fusion_lr=optimizer1.param_groups[0]['lr'],
                    detect_lr=optimizer2.param_groups[0]['lr'],
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_ssim=loss_ssim.item(),
                    loss_fusion=loss_fusion,
                    loss_giou=loss_giou.item(),
                    loss_cls=loss_cls.item(),
                    loss_conf=loss_conf.item(),
                    loss_detect=loss_detect,
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)

        fusion_model_file = os.path.join('weight', 'fusion_model_backup_epoch%g_stage3.pt' % epo)
        torch.save(fusionmodel.state_dict(), fusion_model_file)
        detect_model_file = os.path.join('weight', 'detect_model_backup_epoch%g_stage3.pt' % epo)
        torch.save(detect_model.state_dict(), detect_model_file)

        mAP = 0
        if epo >= 0:
            print('*' * 20 + "Validate" + '*' * 20)
            with torch.no_grad():
                APs = My_Evaluator(fusionmodel,detect_model).APs_voc()
                for i in APs:
                    print("{} --> mAP : {}".format(i, APs[i]))
                    mAP += APs[i]
                mAP = mAP / cfg.DATA["NUM"]
                print('mAP:%g' % (mAP))

            if mAP > best_mAP:
                best_mAP = mAP

            if  cfg.data_type == 'M3FD':
                best_detect_weight = os.path.join('weight', "best_detect_stage3_%g_%g_%g_%g_%g_%g_%g_%g.pt " % (
                    epo, APs['People'], APs['Car'], APs['Bus'], APs['Motorcycle'], APs['Truck'], APs['Lamp'], mAP))
                best_fusion_weight = os.path.join('weight', "best_fusion_stage3_%g_%g_%g_%g_%g_%g_%g_%g.pt " % (
                    epo, APs['People'], APs['Car'], APs['Bus'], APs['Motorcycle'], APs['Truck'], APs['Lamp'], mAP))
            elif cfg.data_type == 'airport':
                best_detect_weight = os.path.join('weight', "best_detection_stage3_%g_%g_%g_%g_%g.pt " % (epo, APs['airplane'], APs['car'], APs['man'], mAP))
                best_fusion_weight = os.path.join('weight', "best_fusion_stage3_%g_%g_%g_%g_%g.pt " % (epo, APs['airplane'], APs['car'], APs['man'], mAP))

            if best_mAP == mAP:
                torch.save(detect_model.state_dict(), best_detect_weight)
                torch.save(fusionmodel.state_dict(), best_fusion_weight)

            print('best mAP : %g' % (best_mAP))

    torch.save(fusionmodel.state_dict(), end2end_fusion_model_name)
    torch.save(detect_model.state_dict(), end2end_detect_model_name)

    logger.info("Fusion Model Save to: {}".format(end2end_fusion_model_name))
    logger.info("Detect Model Save to: {}".format(end2end_detect_model_name))
    logger.info('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--gpu', '-G', type=int, default=0)
    args = parser.parse_args()

    logpath='./logs'
    logger = logging.getLogger()
    setup_logger(logpath)

    train_end2end_fusion(logger)
    # run_fusion('test')
    train_end2end_fusion_detect(logger)
    train_end2end(logger)

