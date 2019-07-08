"""Adapted from https://github.com/JiaRenChang/PSMNet"""
from __future__ import print_function
import argparse
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import skimage
import skimage.io
import skimage.transform

import oc_stereo
from oc_stereo.core import config_utils, constants
from oc_stereo.dataloader.kitti import instance_utils, calib_utils
from oc_stereo.dataloader.kitti.kitti_dataset import KittiDataset
from oc_stereo.models import local

default_config_path = oc_stereo.root_dir() + '/configs/default_train_config.yaml'

data_split_test = 'val_matching_0.5'
default_pretrained_model = os.path.expanduser(
    '~/oc_stereo/data/trained/finetune_12.tar')


# Load in arguments
parser = argparse.ArgumentParser(description='PSMNet')

parser.add_argument('--model',
                    default='local',
                    help='select model')

parser.add_argument('--datatype',
                    default='object',
                    help='datapath')

parser.add_argument('--loadmodel',
                    default=default_pretrained_model,
                    help='load model')

parser.add_argument('--savemodel',
                    default=os.path.expanduser('~/oc_stereo/data/outputs/'),
                    help='save model')

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='enables CUDA training')

parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--config_path',
                    type=str,
                    dest='config_path',
                    default=default_config_path,
                    help='Path to the config')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load in config
config_path = args.config_path
train_config = config_utils.parse_yaml_config(config_path)
val_config = config_utils.parse_yaml_config(config_path)

val_dataset_config = val_config.dataset_config
val_dataset_config.data_split = data_split_test
roi_size = val_dataset_config.roi_size
disp_scale_factor = val_dataset_config.disp_scale_factor
det_type = val_dataset_config.det_type

kitti_test_dataset = KittiDataset(val_dataset_config, 'test')

TestImgLoader = torch.utils.data.DataLoader(
    kitti_test_dataset,
    batch_size=1, shuffle=False, num_workers=12, drop_last=False)

if args.model == 'local':
    model = local.PSMNet(val_dataset_config.disp_range)
else:
    raise ValueError('Invalid model', args.model)

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def get_avg_disp(disp_map, inst_mask):
    return np.mean(disp_map[inst_mask.astype(bool)])


def inference(sample_dict):
    model.eval()

    # Remove incorrect batch of 1 (batched in kitti_dataset.py already)
    sample_name = sample_dict[constants.SAMPLE_NAME]
    num_objs = sample_dict[constants.SAMPLE_NUM_OBJS]
    i2_primes = sample_dict[constants.SAMPLE_I2_PRIME].squeeze(dim=0)
    coord_u_grids = sample_dict[constants.SAMPLE_COORD_GRID_U].squeeze(dim=0)
    img_2_shape = sample_dict[constants.SAMPLE_IMAGE_2_SHAPE]
    img_2_input = sample_dict[constants.SAMPLE_IMAGE_2_INPUT].squeeze(dim=0)
    img_3_input = sample_dict[constants.SAMPLE_IMAGE_3_INPUT].squeeze(dim=0)
    full_img_2_input = sample_dict[constants.SAMPLE_FULL_IMAGE_2_INPUT]
    full_img_3_input = sample_dict[constants.SAMPLE_FULL_IMAGE_3_INPUT]
    rois_left = sample_dict[constants.SAMPLE_LABEL_ROIS_LEFT].squeeze(dim=0)
    rois_right = sample_dict[constants.SAMPLE_LABEL_ROIS_RIGHT].squeeze(dim=0)
    boxes_2d_left = sample_dict[constants.SAMPLE_LABEL_BOXES_2D_LEFT].squeeze(dim=0)
    boxes_2d_right = sample_dict[constants.SAMPLE_LABEL_BOXES_2D_RIGHT].squeeze(dim=0)

    if args.cuda:
        imgL = torch.FloatTensor(img_2_input).cuda()
        imgR = torch.FloatTensor(img_3_input).cuda()
        full_imgL = torch.FloatTensor(full_img_2_input).cuda()
        full_imgR = torch.FloatTensor(full_img_3_input).cuda()
        rois_left = torch.FloatTensor(rois_left).cuda()
        rois_right = torch.FloatTensor(rois_right).cuda()
        boxes_2d_left = torch.FloatTensor(boxes_2d_left).cuda()
        boxes_2d_right = torch.FloatTensor(boxes_2d_right).cuda()

    imgL, imgR, full_imgL, full_imgR, boxes_2d_left, boxes_2d_right, rois_left, rois_right, = \
        Variable(imgL), Variable(imgR), Variable(full_imgL), Variable(full_imgR),\
        Variable(boxes_2d_left), Variable(boxes_2d_right), Variable(rois_left), Variable(rois_right)

    with torch.no_grad():
        inference_start_time = time.time()
        output, instance_masks = model(imgL, imgR, full_imgL, full_imgR, rois_left, rois_right,
                                       kitti_test_dataset.full_img_downsample_scale)

    num_objs = num_objs.data.cpu()
    i2_primes = i2_primes.data.cpu().numpy()
    coord_u_grids = coord_u_grids.data.cpu().numpy()
    img_2_shape = np.array(img_2_shape)
    boxes_2d_left = boxes_2d_left.data.cpu().numpy()
    boxes_2d_right = boxes_2d_right.data.cpu().numpy()
    pred_disps_local = output.data.cpu().numpy()
    instance_masks = instance_masks.data.cpu().numpy()

    instance_masks = (instance_masks > 0.0).astype(np.float32)

    if num_objs == 1:
        instance_masks = np.expand_dims(instance_masks, 0)

    # Remove scaling
    pred_disps_local = pred_disps_local / disp_scale_factor

    pred_global_disp = [instance_utils.calc_global_from_local_disp(
        pred_disp_local, coord_u_grid, i2_prime, box_2d_right, roi_size, instance_mask)
        for (box_2d_right, pred_disp_local, coord_u_grid, i2_prime, instance_mask)
        in zip(boxes_2d_right, pred_disps_local, coord_u_grids, i2_primes, instance_masks)]

    pred_global_disp = np.stack(pred_global_disp, axis=0)

    # Create disp map placeholder
    final_disp_map = np.zeros(img_2_shape, dtype=np.float32)

    # Calculate depth ordering
    depth_ordering = np.argsort(
        [get_avg_disp(disp_map, inst_mask)
         for disp_map, inst_mask in zip(pred_global_disp, instance_masks)])

    # Paste back into original disp map
    for idx in depth_ordering:
        predicted_disp_obj = pred_global_disp[idx, :, :]
        instance_mask = instance_masks[idx, :, :]
        left_box = np.int32(boxes_2d_left[idx, :])
        left_box_height = left_box[2] - left_box[0]
        left_box_width = left_box[3] - left_box[1]

        # Resize to left box size
        resized_pred_disp = cv2.resize(predicted_disp_obj, tuple([left_box_width, left_box_height]),
                                       interpolation=cv2.INTER_NEAREST)

        resized_instance_mask = cv2.resize(instance_mask,
                                           tuple([left_box_width, left_box_height]),
                                           interpolation=cv2.INTER_NEAREST)

        # Mask pred disp
        masked_resized_pred_disp = resized_pred_disp * resized_instance_mask

        # Replace pixels
        local_inst_pixels_mask = np.where(resized_instance_mask)
        final_inst_pixels = final_disp_map[left_box[0]:left_box[2], left_box[1]:left_box[3]]
        final_inst_pixels[local_inst_pixels_mask] = masked_resized_pred_disp[local_inst_pixels_mask]

    inference_time = time.time() - inference_start_time

    return final_disp_map, inference_time


def main():

    # Make output folder
    output_dir = os.path.join(
        args.savemodel,
        train_config.config_name,
        'output_disps')
    if os.path.isdir(output_dir):
        pass
    else:
        os.makedirs(output_dir)

    inference_times = []
    for idx, sample_dict in enumerate(TestImgLoader):

        if not sample_dict:
            continue

        sample_name = sample_dict[constants.SAMPLE_NAME][0]

        start_time = time.time()
        pred_disp, inference_time = inference(sample_dict)

        sys.stdout.write('\r{} / {}'.format(idx + 1, len(TestImgLoader)))
        print(' time = %.2f' % (time.time() - start_time))

        skimage.io.imsave(
            output_dir + '/' + sample_name + '.png', (pred_disp * 256).astype('uint16'))

        inference_times.append(inference_time)

    mean_inference_time = np.round(np.mean(inference_times), 5)
    print('Mean inference time:', mean_inference_time)
    with open(output_dir + '/inference_time.txt', 'w') as f:
        f.write(str(mean_inference_time))


if __name__ == '__main__':
    main()
