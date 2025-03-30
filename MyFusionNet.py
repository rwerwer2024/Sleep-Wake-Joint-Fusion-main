# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)  # 点乘（即“ * ”） ---- 各个矩阵对应元素做乘法;  区别于矩阵乘  .dot（）


class drf_block3(nn.Module):
    def __init__(self,in_channels,out_channels,G=4):
        super(drf_block3, self).__init__()
        self.features = out_channels//4

        self.convs_layer0 = Conv1(in_channels, self.features)

        self.convs_layer1 = nn.Sequential(
            nn.Conv2d(self.features , self.features, kernel_size=3, stride=1, padding=1, dilation=1, groups=G,
                      bias=False),
            nn.BatchNorm2d(self.features),
            nn.ReLU(inplace=False)
        )
        self.convs_layer2 = nn.Sequential(
            nn.Conv2d(2*self.features, self.features, kernel_size=3, stride=1, padding=2, dilation=2, groups=G,
                      bias=False),
            nn.BatchNorm2d(self.features),
            nn.ReLU(inplace=False),
        )
        self.convs_layer3 = nn.Sequential(
            nn.Conv2d(3*self.features, self.features, kernel_size=3, stride=1, padding=3, dilation=3, groups=G,
                      bias=False),
            nn.BatchNorm2d(self.features),
            nn.ReLU(inplace=False)
        )

        self.se = SELayer(4*self.features, 8)

        self.convs_layer4 = Conv1(4*self.features,out_channels)

    def forward(self,x):

        x = self.convs_layer0(x)

        x1 = torch.cat((x,self.convs_layer1(x)),dim=1)
        x2 = torch.cat((x1,self.convs_layer2(x1)), dim=1)
        x3 = torch.cat((x2, self.convs_layer3(x2)), dim=1)
        #feats = torch.cat((x1,x2,x3), dim=1)
        feats_U = self.se(x3)

        feats_U = self.convs_layer4(feats_U)
        # feats = feats.view(batch_size, 3, self.features, feats.shape[2], feats.shape[3])
        # feats_U = torch.sum(feats, dim=1)
        return feats_U



class MyFusionNet3(nn.Module):
    def __init__(self, output):
        super(MyFusionNet3, self).__init__()
        vis_ch = [16,16,16,16]
        inf_ch = [16,16,16,16]

        # vis_ch = [16,32,48,64]
        # inf_ch = [16,32,48,64]

        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1 = drf_block3(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = drf_block3(vis_ch[1], vis_ch[2])
        self.vis_rgbd3 = drf_block3(vis_ch[2], vis_ch[3])

        self.inf_conv = ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = drf_block3(inf_ch[0], inf_ch[1])
        self.inf_rgbd2 = drf_block3(inf_ch[1], inf_ch[2])
        self.inf_rgbd3 = drf_block3(inf_ch[2], inf_ch[3])

        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[3],vis_ch[2])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[2],vis_ch[1])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[1], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], output)

        self.vis_weight1 = torch.nn.Conv2d(vis_ch[1], 1, kernel_size=1, stride=1, padding=0,
                                           bias=True)
        self.vis_weight2 = torch.nn.Conv2d(vis_ch[2] , 1, kernel_size=1, stride=1, padding=0,
                                           bias=True)
        self.vis_weight3 = torch.nn.Conv2d(vis_ch[3],1, kernel_size=1,stride=1, padding=0,
                                          bias=True)

    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p = self.vis_conv(x_vis_origin)
        x_vis_p1 = self.vis_rgbd1(x_vis_p)
        x_vis_p2 = self.vis_rgbd2(x_vis_p1)
        x_vis_p3 = self.vis_rgbd3(x_vis_p2)

        x_inf_p = self.inf_conv(x_inf_origin)
        x_inf_p1 = self.inf_rgbd1(x_inf_p)
        x_inf_p2 = self.inf_rgbd2(x_inf_p1)
        x_inf_p3 = self.inf_rgbd3(x_inf_p2)

        add_weight1 = torch.sigmoid(self.vis_weight1(x_vis_p1+x_inf_p1))
        fusion_img1  = add_weight1 * x_vis_p1 + (1 - add_weight1) * x_inf_p1

        add_weight2 = torch.sigmoid(self.vis_weight2(x_vis_p2+x_inf_p2))
        fusion_img2  = add_weight2 * x_vis_p2 + (1 - add_weight2) * x_inf_p2

        add_weight3 = torch.sigmoid(self.vis_weight3(x_vis_p3+x_inf_p3))
        fusion_img3  = add_weight3 * x_vis_p3 + (1 - add_weight3) * x_inf_p3

        x=self.decode4(fusion_img3)
        x=self.decode3(fusion_img2+x)
        x=self.decode2(fusion_img1+x)
        x=self.decode1(x)
        return x


def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2,4,480,640).astype(np.float32))
    model = MyFusionNet3(output=1)
    y = model(x[:, 0:3, :, :],x[:, 3:4, :, :])
    print('output shape:', y.shape)
    assert y.shape == (2,1,480,640), 'output shape (2,1,480,640) is expected!'
    print('test ok!')

if __name__ == '__main__':
    unit_test()
