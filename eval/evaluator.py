import config.config_voc as cfg
import os
import shutil
from eval import voc_eval
# from utils.datasets import *

from utils.tools import *
from tqdm import tqdm
from utils.visualize import *
from torch.nn import functional as F
from torch.autograd import Variable

import time

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


class Evaluator(object):
    def __init__(self, model, visiual=True):
        self.classes = cfg.DATA["CLASSES"]
        if cfg.data_type == 'voc':
            self.val_data_path = os.path.join(cfg.DATA_PATH, 'VOC2007')
        else:
            self.val_data_path = cfg.DATA_PATH
        self.pred_result_path = os.path.join(cfg.PROJECT_PATH, 'data', 'results')
        self.conf_thresh = cfg.TEST["CONF_THRESH"]
        self.nms_thresh = cfg.TEST["NMS_THRESH"]
        self.val_shape =  cfg.TEST["TEST_IMG_SIZE"]

        self.__visiual = visiual
        self.__visual_imgs = 0

        self.model = model
        self.device = next(model.parameters()).device

    def APs_voc(self, multi_test=False, flip_test=False):
        img_inds_file = os.path.join(self.val_data_path,  'ImageSets', 'Main', 'test.txt')
        with open(img_inds_file, 'r') as f:
            lines = f.readlines()
            img_inds = [line.strip() for line in lines]

        if os.path.exists(self.pred_result_path):
            shutil.rmtree(self.pred_result_path)
        os.mkdir(self.pred_result_path)

        for img_ind in tqdm(img_inds):
            #img_path = os.path.join(self.val_data_path, 'vi', img_ind+'.png')
            if cfg.data_type=="M3FD":
                img_path = os.path.join('./MSRS/Fusion', 'test', cfg.data_type, img_ind+'.png')
                org_img_path = os.path.join('D:/M3FD/M3FD_Detection/vi', img_ind + '.png')
                img = cv2.imread(img_path)
                img_org = cv2.imread(org_img_path)
                img_shape = img_org.shape[:2]

            elif cfg.data_type=="airport":
                img_path = os.path.join('./MSRS/Fusion', 'test', cfg.data_type, img_ind+'.jpg')
                org_img_path = os.path.join('D:/DL/tan/new_airport/JPEGImages/', img_ind + '.jpg')
                img = cv2.imread(img_path)
                img_org = cv2.imread(org_img_path)
                img_shape = img_org.shape[:2]

            #print(img_path)
            bboxes_prd = self.get_bbox(img, img_shape, multi_test, flip_test)

            if bboxes_prd.shape[0]!=0 and self.__visiual and self.__visual_imgs < 100:
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]

                visualize_boxes(image=img_org, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.classes)
                path = os.path.join(cfg.PROJECT_PATH, "data/results/{}.jpg".format(self.__visual_imgs))
                cv2.imwrite(path, img_org)

                self.__visual_imgs += 1

            for bbox in bboxes_prd:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])

                class_name = self.classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                s = ' '.join([img_ind, score, xmin, ymin, xmax, ymax]) + '\n'
                s1 = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(s)

                with open(os.path.join(self.pred_result_path, img_ind + '.txt'), 'a') as f:
                    f.write(s1)

        return self.__calc_APs()

    def get_bbox(self, img, img_shape ,multi_test=False, flip_test=False):
        if multi_test:
            test_input_sizes = range(320, 960, 64)
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale =(0, np.inf)
                bboxes_list.append(self.__predict(img, img_shape, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(img[:, ::-1], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.__predict(img, img_shape ,self.val_shape, (0, np.inf))

        bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)

        # bboxes = nms_new(bboxes, self.conf_thresh, self.nms_thresh)

        return bboxes

    def __predict(self, img, img_shape ,test_shape, valid_scale):
        org_img = np.copy(img)

        org_h, org_w = img_shape[0],img_shape[1]

        # try:
        #     org_h, org_w, _ = org_img.shape
        # except:
        #     print('err')

        img = self.__get_img_tensor(img, test_shape).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, p_d = self.model(img)
        pred_bbox = p_d.squeeze().cpu().numpy()
        bboxes = self.__convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)

        return bboxes

    def __get_img_tensor(self, img, test_shape):
        img, _ , _ = transfrom.Resize(test_shape)(np.copy(img))
        # img, _ , _ = transfrom.SubtractMeans([0, 0, 0] )(np.copy(img))  # 减均值
        # img, _ , _ = transfrom.DivideStds([1, 1, 1] )(np.copy(img))  # 除方差
        # img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...].transpose(0, 3, 1, 2)).float()


    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        """
        预测框进行过滤，去除尺度不合理的框
        """
        pred_coor = xywh2xyxy(pred_bbox[:, :4])
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
        # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
        org_h, org_w = org_img_shape
        # resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        # dw = (test_input_size - resize_ratio * org_w) / 2
        # dh = (test_input_size - resize_ratio * org_h) / 2
        # pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        # pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        pred_coor[:, 0::2] *= org_w
        pred_coor[:, 1::2] *= org_h
        # (2)将预测的bbox中超出原图的部分裁掉
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        # (3)将无效bbox的coor置为0
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # (4)去掉不在有效范围内的bbox
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # (5)将score低于score_threshold的bbox去掉
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        return bboxes


    def __calc_APs(self, iou_thresh=0.5, use_07_metric=False):
        """
        计算每个类别的ap值
        :param iou_thresh:
        :param use_07_metric:
        :return:dict{cls:ap}
        """
        filename = os.path.join(self.pred_result_path, 'comp4_det_test_{:s}.txt')
        cachedir = os.path.join(self.pred_result_path, 'cache')
        annopath = os.path.join(self.val_data_path, 'Annotations\{:s}.xml')
        imagesetfile = os.path.join(self.val_data_path,  'ImageSets', 'Main', 'test.txt')
        APs = {}
        for i, cls in enumerate(self.classes):
            R, P, AP = voc_eval.voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thresh, use_07_metric)
            APs[cls] = AP
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)

        return APs



class My_Evaluator(object):
    def __init__(self, fusion_model,detect_model, visiual=True):
        self.classes = cfg.DATA["CLASSES"]
        if cfg.data_type == 'voc':
            self.val_data_path = os.path.join(cfg.DATA_PATH, 'VOC2007')
        else:
            self.val_data_path = cfg.DATA_PATH
        self.pred_result_path = os.path.join(cfg.PROJECT_PATH, 'data', 'results')
        self.conf_thresh = cfg.TEST["CONF_THRESH"]
        self.nms_thresh = cfg.TEST["NMS_THRESH"]
        self.val_shape =  cfg.TEST["TEST_IMG_SIZE"]

        self.__visiual = visiual
        self.__visual_imgs = 0
        self.__visual_imgs_1 = 0

        self.fusion_model = fusion_model
        self.detect_model = detect_model
        self.device = next(detect_model.parameters()).device

    def APs_voc(self, multi_test=False, flip_test=False):
        img_inds_file = os.path.join(self.val_data_path,  'ImageSets', 'Main', 'test.txt')
        with open(img_inds_file, 'r') as f:
            lines = f.readlines()
            img_inds = [line.strip() for line in lines]

        if os.path.exists(self.pred_result_path):
            shutil.rmtree(self.pred_result_path)
        os.mkdir(self.pred_result_path)

        for img_ind in tqdm(img_inds):

            vi_img_path = os.path.join('J:/tan/M3FD/M3FD_Detection/vi', img_ind + '.png')
            ir_img_path = os.path.join('J:/tan/M3FD/M3FD_Detection/ir', img_ind + '.png')

            image_vis = cv2.imread(vi_img_path, 1)
            image_ir = cv2.imread(ir_img_path, 0)
            img_shape = image_ir.shape[:2]
            image_vis = cv2.resize(image_vis, (cfg.TRAIN["TRAIN_IMG_SIZE"], cfg.TRAIN["TRAIN_IMG_SIZE"]))
            image_ir = cv2.resize(image_ir, (cfg.TRAIN["TRAIN_IMG_SIZE"], cfg.TRAIN["TRAIN_IMG_SIZE"]))

            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0) / 255.0

            image_vis = Variable(torch.tensor(image_vis[np.newaxis, ...]).float()).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(torch.tensor(image_ir[np.newaxis, ...]).float()).cuda()
            logits = self.fusion_model(image_vis_ycrcb, image_ir)
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
            # fused_image = fused_image.transpose((0, 2, 3, 1))
            fusion_image = (fusion_image - torch.min(fusion_image)) / (
                    torch.max(fusion_image) - torch.min(fusion_image)
            )

            fusion_image = torch.round((255.0 * fusion_image)).float()
            fusion_image_BGR = fusion_image[:,[2,1,0],:,:]


            # ###############################################
            if self.__visual_imgs_1<100:
                fusion_image = fusion_image_BGR
                fused_dir = os.path.join('./MSRS/Fusion', 'train1', cfg.data_type)
                for k in range(len(image_vis)) :
                    image = np.uint8(fusion_image[k, :, :, :].cpu().numpy())
                    image = image.squeeze()
                    image = image.transpose((1, 2, 0))
                    image = Image.fromarray(image)
                    save_path = os.path.join(fused_dir, img_ind + '.png')
                    image.save(save_path)
                    time.sleep(0.01)
                    show_img=cv2.resize(cv2.imread(save_path),(img_shape[1],img_shape[0]))
                    time.sleep(0.01)
            self.__visual_imgs_1 +=1
            # ###############################################

            img = fusion_image_BGR
            #print(img_path)
            bboxes_prd = self.get_bbox(img, img_shape, multi_test, flip_test)

            if bboxes_prd.shape[0]!=0 and self.__visiual and self.__visual_imgs < 100:
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]

                visualize_boxes(image=show_img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.classes)
                path = os.path.join(cfg.PROJECT_PATH, "data/results/{}.jpg".format(img_ind))
                cv2.imwrite(path, show_img)
                time.sleep(0.01)
                self.__visual_imgs += 1

            for bbox in bboxes_prd:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])

                class_name = self.classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                s = ' '.join([img_ind, score, xmin, ymin, xmax, ymax]) + '\n'
                s1 = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(s)

                with open(os.path.join(self.pred_result_path, img_ind + '.txt'), 'a') as f:
                    f.write(s1)

        return self.__calc_APs()

    def get_bbox(self, img, img_shape ,multi_test=False, flip_test=False):
        if multi_test:
            test_input_sizes = range(320, 960, 64)
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale =(0, np.inf)
                bboxes_list.append(self.__predict(img, img_shape, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(img[:, ::-1], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.__predict(img, img_shape ,self.val_shape, (0, np.inf))

        bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)

        # bboxes = nms_new(bboxes, self.conf_thresh, self.nms_thresh)

        return bboxes

    def __predict(self, img, img_shape ,test_shape, valid_scale):
        # org_img = np.copy(img)

        org_h, org_w = img_shape[0],img_shape[1]

        # try:
        #     org_h, org_w, _ = org_img.shape
        # except:
        #     print('err')

        # img = self.__get_img_tensor(img, test_shape).to(self.device)
        self.detect_model.eval()
        with torch.no_grad():
            _, p_d = self.detect_model(img)
        pred_bbox = p_d.squeeze().cpu().numpy()
        bboxes = self.__convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)

        return bboxes

    def __get_img_tensor(self, img, test_shape):
        img, _ , _ = transfrom.Resize(test_shape)(img)
        img = transfrom.Resize(test_shape)(img)
        # img, _ , _ = transfrom.SubtractMeans([0, 0, 0] )(np.copy(img))  # 减均值
        # img, _ , _ = transfrom.DivideStds([1, 1, 1] )(np.copy(img))  # 除方差
        # img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(2, 0, 1)
        # return torch.from_numpy(img[np.newaxis, ...].transpose(0, 3, 1, 2)).float()

        return img


    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        """
        预测框进行过滤，去除尺度不合理的框
        """
        pred_coor = xywh2xyxy(pred_bbox[:, :4])
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
        # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
        org_h, org_w = org_img_shape
        # resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        # dw = (test_input_size - resize_ratio * org_w) / 2
        # dh = (test_input_size - resize_ratio * org_h) / 2
        # pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        # pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        pred_coor[:, 0::2] *= org_w
        pred_coor[:, 1::2] *= org_h
        # (2)将预测的bbox中超出原图的部分裁掉
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        # (3)将无效bbox的coor置为0
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # (4)去掉不在有效范围内的bbox
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # (5)将score低于score_threshold的bbox去掉
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        return bboxes


    def __calc_APs(self, iou_thresh=0.5, use_07_metric=False):
        """
        计算每个类别的ap值
        :param iou_thresh:
        :param use_07_metric:
        :return:dict{cls:ap}
        """
        filename = os.path.join(self.pred_result_path, 'comp4_det_test_{:s}.txt')
        cachedir = os.path.join(self.pred_result_path, 'cache')
        annopath = os.path.join(self.val_data_path, 'Annotations\{:s}.xml')
        imagesetfile = os.path.join(self.val_data_path,  'ImageSets', 'Main', 'test.txt')
        APs = {}
        for i, cls in enumerate(self.classes):
            R, P, AP = voc_eval.voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thresh, use_07_metric)
            APs[cls] = AP
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)

        return APs
