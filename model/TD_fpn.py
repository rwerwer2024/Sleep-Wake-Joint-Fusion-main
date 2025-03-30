import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_module import Convolutional


class FPN_org(nn.Module):
    def __init__(self, fileters_in, fileters_out):
        super(FPN_org, self).__init__()

        feature_size = 256
        C5_size,C4_size,C3_size,C2_size = fileters_in
        fo_0, fo_1, fo_2, fo_3 = fileters_out

        # Smooth layers
        self.smooth0 = nn.Conv2d(256, fo_0, kernel_size=3, stride=1, padding=1)
        self.smooth1 = nn.Conv2d(256, fo_1, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, fo_2, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, fo_3, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer0 = nn.Conv2d(C5_size, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(C4_size, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(C3_size, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(C2_size, 256, kernel_size=1, stride=1, padding=0)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self,  c5, c4, c3, c2):
        # Top-down
        p5 = self.latlayer0(c5)
        print(f'p5:{p5.shape}')
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        print(f'latlayer1(c4):{self.latlayer1(c4).shape}, p4:{p4.shape}')

        p3 = self._upsample_add(p4, self.latlayer2(c3))
        print(f'latlayer1(c3):{self.latlayer2(c3).shape}, p3:{p3.shape}')

        p2 = self._upsample_add(p3, self.latlayer3(c2))
        print(f'latlayer1(c2):{self.latlayer3(c2).shape}, p2:{p2.shape}')

        # Smooth
        p5 = self.smooth0(p5)
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5


class Retinanet_PyramidFeatures(nn.Module):
    def __init__(self,fileters_in, fileters_out):
        super(Retinanet_PyramidFeatures, self).__init__()

        feature_size = 256
        C5_size,C4_size,C3_size = fileters_in
        fo_0, fo_1, fo_2 = fileters_out

        # upsample C5 to get P5 from the FPN paper
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P5_OUT = nn.Sequential(
            Convolutional(filters_in=feature_size, filters_out=2*feature_size, kernel_size=3, stride=1,pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=2*feature_size, filters_out=fo_0, kernel_size=1,stride=1, pad=0)
        )

        # add P5 elementwise to C4
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_1 = nn.Conv2d(C4_size+C5_size, 2*feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(2*feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_OUT = nn.Sequential(
            Convolutional(filters_in=feature_size, filters_out=2*feature_size, kernel_size=3, stride=1,pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=2*feature_size, filters_out=fo_1, kernel_size=1,stride=1, pad=0)
        )

        # add P4 elementwise to C3
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_1 = nn.Conv2d(C3_size+C4_size+C5_size, 2*feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(2*feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_OUT = nn.Sequential(
            Convolutional(filters_in=feature_size, filters_out=2*feature_size, kernel_size=3, stride=1,pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=2*feature_size, filters_out=fo_2, kernel_size=1,stride=1, pad=0)
        )


    def forward(self, C5, C4, C3):

        P5_upsampled_x = self.P5_upsampled(C5)
        P5_x = self.P5_1(C5)
        P5_x = self.P5_2(P5_x)
        P5_x = self.P5_OUT(P5_x)

        P4_x = torch.cat([P5_upsampled_x, C4], 1)
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_1(P4_x)
        P4_x = self.P4_2(P4_x)
        P4_x = self.P5_OUT(P4_x)

        P3_x = torch.cat([P4_upsampled_x, C3], 1)
        # P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_1(P3_x)
        P3_x = self.P3_2(P3_x)
        P3_x = self.P5_OUT(P3_x)


        return [P3_x, P4_x, P5_x]


class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)



class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        """
        x1 means previous output; x2 means current output
        """
        # out = torch.add((x2, x1), dim=1)
        out = torch.cat((x2, x1), dim=1)
        return out


class FPN_YOLOV3(nn.Module):
    """
    FPN for yolov3, and is different from original FPN or retinanet' FPN.
    """
    def __init__(self, fileters_in, fileters_out):
        super(FPN_YOLOV3, self).__init__()

        fi_0, fi_1, fi_2 = fileters_in
        fo_0, fo_1, fo_2 = fileters_out

        # large
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1,pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv0_0 = Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv0_1 = Convolutional(filters_in=1024, filters_out=fo_0, kernel_size=1,
                                       stride=1, pad=0)


        self.__conv0 = Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                                      activate="leaky")
        self.__upsample0 = Upsample(scale_factor=2)
        self.__route0 = Route()

        # medium
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fi_1+256, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv1_0 = Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv1_1 = Convolutional(filters_in=512, filters_out=fo_1, kernel_size=1,
                                       stride=1, pad=0)


        self.__conv1 = Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                                     activate="leaky")
        self.__upsample1 = Upsample(scale_factor=2)
        self.__route1 = Route()

        # small
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fi_2+128, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv2_0 = Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv2_1 = Convolutional(filters_in=256, filters_out=fo_2, kernel_size=1,
                                       stride=1, pad=0)

    def forward(self, x0, x1, x2):  # large, medium, small
        # large
        r0 = self.__conv_set_0(x0)
        out0 = self.__conv0_0(r0)
        out0 = self.__conv0_1(out0)

        # medium
        r1 = self.__conv0(r0)
        r1 = self.__upsample0(r1)
        x1 = self.__route0(x1, r1)
        r1 = self.__conv_set_1(x1)
        out1 = self.__conv1_0(r1)
        out1 = self.__conv1_1(out1)

        # small
        r2 = self.__conv1(r1)
        r2 = self.__upsample1(r2)
        x2 = self.__route1(x2, r2)
        r2 = self.__conv_set_2(x2)
        out2 = self.__conv2_0(r2)
        out2 = self.__conv2_1(out2)

        return out2, out1, out0  # small, medium, large



class FPN_YOLOV3_new_13(nn.Module):
    """
    FPN for yolov3, and is different from original FPN or retinanet' FPN.
    """
    def __init__(self, fileters_in, fileters_out):
        super(FPN_YOLOV3_new_13, self).__init__()

        fi_0, fi_1, fi_2, fi_3 = fileters_in  #2048,1024,512,256
        fo_0, fo_1, fo_2, fo_3 = fileters_out

        # large
        self.__conv5_0 = nn.Sequential(
            Convolutional(filters_in=fi_0, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",groups=8,
                          activate="leaky")
        )

        self.__conv5_out1 = nn.Sequential(
            # Convolutional(filters_in=256, filters_out=256, kernel_size=3, stride=1,groups=8,
            #                            pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=256, filters_out=fo_0, kernel_size=1,
                                       stride=1, pad=0))

        self.__conv5_upsample = nn.Sequential(
            Upsample(scale_factor=2))


        # medium
        self.__conv4_0 = nn.Sequential(
            Convolutional(filters_in=fi_1, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",groups=8,
                          activate="leaky"),
        )
        self.__conv4_out1 = nn.Sequential(
            # Convolutional(filters_in=256, filters_out=256, kernel_size=3, stride=1,groups=8,
            #                            pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=256, filters_out=fo_1, kernel_size=1,
                                       stride=1, pad=0))


        self.__conv4_upsample = nn.Sequential(
            Upsample(scale_factor=2))

        # self.__route4 = Route()

        # small
        self.__conv3_0 = nn.Sequential(
            Convolutional(filters_in=fi_2, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",groups=8,
                          activate="leaky")
        )
        self.__conv3_out1 = nn.Sequential(
            # Convolutional(filters_in=256, filters_out=256, kernel_size=3, stride=1,groups=8,
            #                            pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=256, filters_out=fo_2, kernel_size=1,
                                       stride=1, pad=0))

        self.__conv3_upsample = nn.Sequential(
            Upsample(scale_factor=2))

        # smaller
        self.__conv2_0 = nn.Sequential(
            Convolutional(filters_in=fi_3, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",groups=8,
                          activate="leaky")
        )
        self.__conv2_out1 = nn.Sequential(
            # Convolutional(filters_in=256, filters_out=256, kernel_size=3, stride=1,groups=8,
            #                            pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=256, filters_out=fo_3, kernel_size=1,
                                       stride=1, pad=0))

        # self.__route3 = Route()

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                #print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                #print("initing {}".format(m))

    def forward(self, x5, x4, x3, x2):  # large, medium, small

        # large
        r5 = self.__conv5_0(x5)
        out5 = self.__conv5_out1(r5)

        # medium
        r4 = self.__conv4_0(x4) + self.__conv5_upsample(r5)
        out4 = self.__conv4_out1(r4)

        # small
        r3 = self.__conv3_0(x3) + self.__conv4_upsample(r4)
        out3 = self.__conv3_out1(r3)

        # smaller
        r2 = self.__conv2_0(x2) + self.__conv3_upsample(r3)
        out2 = self.__conv2_out1(r2)

        return out2, out3, out4, out5 # small, medium, large

