from __future__ import print_function
import math

# from prroi_pool.functional import prroi_pool2d
from faster_rcnn_pytorch.lib.model.roi_layers import ROIAlign
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from oc_stereo.models.submodule_local import *


class hourglass(nn.Module):

    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_3d(
                inplanes,
                inplanes * 2,
                kernel_size=3,
                stride=2,
                pad=1),
            nn.ReLU(
                inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(
            convbn_3d(
                inplanes * 2,
                inplanes * 2,
                kernel_size=3,
                stride=2,
                pad=1),
            nn.ReLU(
                inplace=True))

        self.conv4 = nn.Sequential(
            convbn_3d(
                inplanes * 2,
                inplanes * 2,
                kernel_size=3,
                stride=1,
                pad=1),
            nn.ReLU(
                inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2,
                inplanes * 2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False),
            nn.BatchNorm3d(
                inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2,
                inplanes,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNet(nn.Module):

    def __init__(self, disp_range):
        super(PSMNet, self).__init__()
        self.max_disp = disp_range[1]
        self.min_disp = disp_range[0]
        self.disp_range = self.max_disp - self.min_disp

        self.feature_extraction_crop = feature_extraction()
        self.feature_extraction_full = feature_extraction()

        self.one_by_one = nn.Conv2d(64, 32, [1, 1])

        self.decoder = Decoder()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(
            convbn_3d(
                32, 32, 3, 1, 1), nn.ReLU(
                inplace=True), nn.Conv3d(
                32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(
            convbn_3d(
                32, 32, 3, 1, 1), nn.ReLU(
                inplace=True), nn.Conv3d(
                32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(
            convbn_3d(
                32, 32, 3, 1, 1), nn.ReLU(
                inplace=True), nn.Conv3d(
                32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right, full_left, full_right, rois_l, rois_r,
                full_img_downsample_scale):

        crop_refimg_fea = self.feature_extraction_crop(left)
        crop_targetimg_fea = self.feature_extraction_crop(right)

        # Feature size
        local_feature_size = crop_refimg_fea.size()[2:4]

        # Get full image features
        full_refimg_fea = self.feature_extraction_full(full_left)
        full_targetimg_fea = self.feature_extraction_full(full_right)

        # Take crops from full image
        roi_align = ROIAlign((local_feature_size[0], local_feature_size[1]),
                             1.0/(4.0 * full_img_downsample_scale), 2)

        crop_imgL = roi_align(full_refimg_fea, rois_l)
        crop_imgR = roi_align(full_targetimg_fea, rois_r)

        # 1x1 conv
        refimg_fea = crop_imgL * crop_refimg_fea
        targetimg_fea = crop_imgR * crop_targetimg_fea

        # Use decoder to get instance mask
        pred_instance_masks = self.decoder(refimg_fea)
        pred_instance_masks = torch.squeeze(pred_instance_masks)

        # Matching
        cost = Variable(
            torch.FloatTensor(
                refimg_fea.size()[0],
                refimg_fea.size()[1] * 2,
                self.disp_range / 4,
                refimg_fea.size()[2],
                refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.disp_range / 4):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.upsample(cost1, [self.disp_range, left.size()[2],
                                       left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.disp_range, left.size()[2],
                                       left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression(self.max_disp, self.min_disp, self.disp_range)(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression(self.max_disp, self.min_disp, self.disp_range)(pred2)

        cost3 = F.upsample(cost3, [self.disp_range, left.size()[2], left.size()[3]],
                           mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        softmax = F.softmax(cost3, dim=1)
        pred3 = disparityregression(self.max_disp, self.min_disp, self.disp_range)(softmax)

        if self.training:
            return pred1, pred2, pred3, pred_instance_masks
        else:
            return pred3, pred_instance_masks
