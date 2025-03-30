import sys
sys.path.append("..")

import torch.nn as nn
from model.TD_fpn import FPN_YOLOV3_new_13
from model.conv_module import Convolutional

from utils.tools import *
import utils.anchors as anchors
from utils.Boxs_op import  convert_locations_to_boxes
from resnet_detect import ResNet, BasicBlock, Bottleneck

import torch.utils.model_zoo as model_zoo
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'se_resnet50': "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl",
}
class TD_Net(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pretrained = True):
        super(TD_Net, self).__init__()
        self.__anchors_per_scale = 5
        self.__nC = cfg.DATA["NUM"]
        self.__out_channel = self.__anchors_per_scale  * (self.__nC + 5)
        self.__anchors_new = anchors.Anchors_4()

        block = Bottleneck
        layers = [3,4,6,3]
        self.__backnone = ResNet(self.__nC, block, layers)

        if block == BasicBlock:
            fpn_sizes = [self._TD_Net__backnone.layer4[layers[3] - 1].conv2.out_channels, self._TD_Net__backnone.layer3[layers[1] - 1].conv2.out_channels,
                         self._TD_Net__backnone.layer2[layers[1] - 1].conv2.out_channels,self._TD_Net__backnone.layer1[layers[0] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self._TD_Net__backnone.layer4[layers[3] - 1].conv3.out_channels, self._TD_Net__backnone.layer3[layers[2] - 1].conv3.out_channels,
                         self._TD_Net__backnone.layer2[layers[1] - 1].conv3.out_channels, self._TD_Net__backnone.layer1[layers[0] - 1].conv3.out_channels]
        self.__fpn = FPN_YOLOV3_new_13(fileters_in=fpn_sizes,
                                       fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel, self.__out_channel])

        if pretrained:
            self.__backnone.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
            self.__backnone.freeze_bn()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        device = x.device
        image_size = x.shape[2:]
        center_form_anchors = self.__anchors_new(image_size)
        x_mini, x_s, x_m, x_l = self.__backnone(x)
        x_mini, x_s, x_m, x_l= self.__fpn(x_l, x_m, x_s, x_mini)

        p = []
        for x in [x_mini, x_s, x_m, x_l]:
            bs, nG = x.shape[0], x.shape[-1]
            x = x.view(bs, self.__anchors_per_scale, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)
            x = x.contiguous().view(bs, -1, 5 + self.__nC)
            p.append(x)

        p = torch.cat(p, dim=1)
        p_de = self.__decode(p.clone(),center_form_anchors.float().to(device))

        return p,p_de


    def __decode(self, p, center_form_anchors):

        conv_raw_dxdydwdh = p[..., 0:4]
        conv_raw_conf = p[..., 4:5]
        conv_raw_prob = p[..., 5:]
        # class_id = torch.argmax(F.softmax(conv_raw_prob, dim=1), dim=-1)
        if type(conv_raw_dxdydwdh) is np.ndarray:
            conv_raw_dxdydwdh = torch.from_numpy(conv_raw_dxdydwdh).float()
        if type(center_form_anchors) is np.ndarray:
            center_form_anchors = torch.from_numpy(center_form_anchors).float()

        center_form_boxes = convert_locations_to_boxes(
            conv_raw_dxdydwdh.contiguous().view(p.shape[0], -1, 4), center_form_anchors
        )

        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        # pred_conf = conv_raw_conf
        # pred_prob = conv_raw_prob
        # pred_prob = conv_raw_prob  # 注意，在yolo loss里面重新经过了softmax,所以此处不需要sigmoid
        pred_bbox = torch.cat([center_form_boxes,pred_conf, pred_prob], dim=-1)

        return pred_bbox.view(p.shape[0], -1, 5 + self.__nC) if not self.training else pred_bbox

    def __init_weights(self):

        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))


    def load_darknet_weights(self, weight_file, cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"

        print("load darknet weights : ", weight_file)

        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1

                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

                print("loading weight {}".format(conv_layer))


if __name__ == '__main__':
    net = TD_Net()
    print(net)

    in_img = torch.randn(12, 3, 416, 416)
    p, p_d = net(in_img)

    for i in range(3):
        print(p[i].shape)
        print(p_d[i].shape)