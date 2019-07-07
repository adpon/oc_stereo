from __future__ import print_function
import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import oc_stereo
from oc_stereo.core import config_utils, constants
from oc_stereo.dataloader.kitti import instance_utils, calib_utils
from oc_stereo.dataloader.kitti.kitti_dataset import KittiDataset
from oc_stereo.models import local


default_config_path = oc_stereo.root_dir() + '/configs/default_train_config.yaml'

data_split_train = 'train'
data_split_test = 'val_matching_0.5_hare'

default_num_epochs = 300

default_pretrained_model = os.path.expanduser(
    '~/oc_stereo/data/trained/finetune_12.tar')


default_load_optimizer = False

# Load in arguments
parser = argparse.ArgumentParser(description='PSMNet')

parser.add_argument('--model',
                    default='local',
                    help='select model')

parser.add_argument('--datatype',
                    default='object',
                    help='datapath')

parser.add_argument('--epochs',
                    type=int,
                    default=default_num_epochs,
                    help='number of epochs to train')

parser.add_argument('--loadmodel',
                    default=default_pretrained_model,
                    help='load model')

parser.add_argument('--load_optimizer',
                    default=default_load_optimizer,
                    help='load optimizer')

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

train_dataset_config = train_config.dataset_config
train_dataset_config.data_split = data_split_train
disp_range = train_dataset_config.disp_range

val_dataset_config = val_config.dataset_config
val_dataset_config.data_split = data_split_test
roi_size = val_dataset_config.roi_size
disp_scale_factor = val_dataset_config.disp_scale_factor

# Make output folder
save_dir = os.path.join(args.savemodel, train_config.config_name)
if os.path.isdir(save_dir):
    pass
else:
    os.makedirs(save_dir)

masks_dir = os.path.join(save_dir, 'masks')
if os.path.isdir(masks_dir):
    pass
else:
    os.makedirs(masks_dir)

kitti_train_dataset = KittiDataset(train_dataset_config, 'train')
kitti_val_dataset = KittiDataset(val_dataset_config, 'val')

# Set up data loaders
TrainImgLoader = torch.utils.data.DataLoader(
    kitti_train_dataset,
    batch_size=1, shuffle=True, num_workers=12, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    kitti_val_dataset,
    batch_size=1, shuffle=False, num_workers=12, drop_last=False)

if args.model == 'local':
    model = local.PSMNet(train_dataset_config.disp_range)
else:
    raise ValueError('Invalid model', args.model)

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    # Load in pretrained weights
    model_dict = model.state_dict()

    pretrained_dict = torch.load(args.loadmodel)

    adapted_pretrained_dict = {}
    # Update pretrained_dict for the two feature extractors
    for key, var in pretrained_dict['state_dict'].items():
        if 'feature_extraction' in key:
            adapted_pretrained_dict[
                key.replace('feature_extraction', 'feature_extraction_crop')] = var
            adapted_pretrained_dict[
                key.replace('feature_extraction', 'feature_extraction_full')] = var
        else:
            adapted_pretrained_dict[key] = var

    adapted_pretrained_dict = {k: v for k, v in adapted_pretrained_dict.items() if k in model_dict}

    model_dict.update(adapted_pretrained_dict)

    model.load_state_dict(model_dict)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

if args.load_optimizer:
    optimizer.load_state_dict(pretrained_dict['optimizer'])

# Set up TensorBoard
log_dir = save_dir + '/logs'
writer = SummaryWriter(log_dir)
if os.path.isdir(log_dir):
    pass
else:
    os.makedirs(log_dir)

use_srgt_masks = train_dataset_config.use_srgt_masks


def train(sample_dict, global_step):
    model.train()

    # Remove incorrect batch of 1 (batched in kitti_dataset.py already)
    img_2_input = sample_dict[constants.SAMPLE_IMAGE_2_INPUT].squeeze()
    img_3_input = sample_dict[constants.SAMPLE_IMAGE_3_INPUT].squeeze()

    full_img_2_input = sample_dict[constants.SAMPLE_FULL_IMAGE_2_INPUT]
    full_img_3_input = sample_dict[constants.SAMPLE_FULL_IMAGE_3_INPUT]

    rois_left = sample_dict[constants.SAMPLE_LABEL_ROIS_LEFT].squeeze()
    rois_right = sample_dict[constants.SAMPLE_LABEL_ROIS_RIGHT].squeeze()

    boxes_2d_left = sample_dict[constants.SAMPLE_LABEL_BOXES_2D_LEFT].squeeze()
    boxes_2d_right = sample_dict[constants.SAMPLE_LABEL_BOXES_2D_RIGHT].squeeze()

    coord_u_grids = sample_dict[constants.SAMPLE_COORD_GRID_U].squeeze()
    i2_primes = sample_dict[constants.SAMPLE_I2_PRIME].squeeze()

    disp_map = sample_dict[constants.SAMPLE_DISP_MAPS_LOCAL].squeeze()
    gt_disp_map_global = sample_dict[constants.SAMPLE_DISP_MAPS_GLOBAL].squeeze()
    instance_masks = sample_dict[constants.SAMPLE_INSTANCE_MASKS].squeeze()

    gt_global_xyz_map = sample_dict[constants.SAMPLE_GT_GLOBAL_XYZ_MAP].squeeze()
    inst_xyz_masks = sample_dict[constants.SAMPLE_INST_XYZ_MASKS].squeeze()

    # Stereo calib parameters
    stereo_calib_f = sample_dict[constants.SAMPLE_STEREO_CALIB_F]
    stereo_calib_b = sample_dict[constants.SAMPLE_STEREO_CALIB_B]
    stereo_calib_center_u = sample_dict[constants.SAMPLE_STEREO_CALIB_CENTER_U]
    stereo_calib_center_v = sample_dict[constants.SAMPLE_STEREO_CALIB_CENTER_V]

    # Set as Variables
    imgL = Variable(torch.FloatTensor(img_2_input))
    imgR = Variable(torch.FloatTensor(img_3_input))
    full_imgL = Variable(torch.FloatTensor(full_img_2_input))
    full_imgR = Variable(torch.FloatTensor(full_img_3_input))

    torch_roi_size = Variable(torch.FloatTensor(roi_size))
    coord_u_grids = Variable(torch.FloatTensor(coord_u_grids))
    i2_primes = Variable(torch.FloatTensor(i2_primes.float()))

    rois_left = Variable(torch.FloatTensor(rois_left))
    rois_right = Variable(torch.FloatTensor(rois_right))

    boxes_2d_left = Variable(torch.FloatTensor(boxes_2d_left))
    boxes_2d_right = Variable(torch.FloatTensor(boxes_2d_right))

    disp_l = Variable(torch.FloatTensor(disp_map.float()))
    global_disp_l = Variable(torch.FloatTensor(gt_disp_map_global.float()))
    instance_masks = Variable(torch.FloatTensor(instance_masks.float()))
    gt_global_xyz_map = Variable(torch.FloatTensor(gt_global_xyz_map.float()))
    inst_xyz_masks = Variable(torch.FloatTensor(inst_xyz_masks.float()))

    stereo_calib_f = Variable(torch.FloatTensor(stereo_calib_f))
    stereo_calib_b = Variable(torch.FloatTensor(stereo_calib_b))
    stereo_calib_center_u = Variable(torch.FloatTensor(stereo_calib_center_u))
    stereo_calib_center_v = Variable(torch.FloatTensor(stereo_calib_center_v))

    if args.cuda:
        imgL, imgR, full_imgL, full_imgR, disp_l, instance_masks, \
            torch_roi_size, coord_u_grids, i2_primes, \
            boxes_2d_left, boxes_2d_right,\
            rois_left, rois_right, \
            stereo_calib_f, stereo_calib_b, stereo_calib_center_u, stereo_calib_center_v,\
            global_disp_l, gt_global_xyz_map, inst_xyz_masks = \
            imgL.cuda(), imgR.cuda(), \
            full_imgL.cuda(), full_imgR.cuda(), \
            disp_l.cuda(), instance_masks.cuda(), torch_roi_size.cuda(), \
            coord_u_grids.cuda(), i2_primes.cuda(), \
            boxes_2d_left.cuda(), boxes_2d_right.cuda(), \
            rois_left.cuda(), rois_right.cuda(), \
            stereo_calib_f.cuda(), stereo_calib_b.cuda(), \
            stereo_calib_center_u.cuda(), stereo_calib_center_v.cuda(), global_disp_l.cuda(), \
            gt_global_xyz_map.cuda(), inst_xyz_masks.cuda()

        if use_srgt_masks:
            srgt_instance_masks = sample_dict[constants.SAMPLE_SRGT_INSTANCE_MASKS].squeeze()
            srgt_instance_masks = Variable(torch.FloatTensor(srgt_instance_masks.float()))
            srgt_instance_masks = srgt_instance_masks.cuda()

    # Create mask
    mask = (disp_l > disp_range[0]) & (disp_l < disp_range[1]) & (disp_l != 0)
    mask.detach_()

    optimizer.zero_grad()

    # Have the same image twice for data parallelism
    full_imgL = torch.cat([full_imgL, full_imgL], 0)
    full_imgR = torch.cat([full_imgR, full_imgR], 0)

    # Local disparity maps
    output1, output2, output3, pred_instance_masks = model(
        imgL, imgR,
        full_imgL, full_imgR,
        rois_left, rois_right,
        kitti_train_dataset.full_img_downsample_scale)

    # Compute losses
    total_loss = 0

    # Convert to global disparity map
    pred_global_disps = [instance_utils.calc_global_from_local_disp(
        pred_disp_local, coord_u_grid, i2_prime, box_2d_right, roi_size, instance_mask)
        for (box_2d_right, pred_disp_local, coord_u_grid, i2_prime, instance_mask)
        in zip(boxes_2d_right, output3, coord_u_grids, i2_primes, instance_masks)]
    pred_global_disps = torch.stack(pred_global_disps, dim=0)

    # Convert disparity to point cloud
    pred_global_disps = pred_global_disps * instance_masks
    pred_xyz = calib_utils.torch_pc_from_disparity(pred_global_disps, stereo_calib_f, stereo_calib_b,
                                                   stereo_calib_center_u, stereo_calib_center_v)

    # XYZ loss
    num_valid_points = torch.sum(instance_masks)
    xyz_loss = F.smooth_l1_loss(pred_xyz, gt_global_xyz_map, reduction='sum') / num_valid_points
    xyz_loss *= 0.0001

    total_loss += xyz_loss
    writer.add_scalar('xyz_loss', xyz_loss, global_step)

    # Disparity loss
    l1_loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_l[mask], size_average=True)\
        + 0.7 * F.smooth_l1_loss(output2[mask], disp_l[mask], size_average=True)\
        + F.smooth_l1_loss(output3[mask], disp_l[mask], size_average=True)

    total_loss += l1_loss
    writer.add_scalar('l1_loss', l1_loss, global_step)

    # Instance mask loss
    if use_srgt_masks:
        gt_instance_masks = srgt_instance_masks
    else:
        gt_instance_masks = instance_masks

    instance_loss = F.binary_cross_entropy_with_logits(pred_instance_masks, gt_instance_masks,
                                                       reduction='none')
    num_pixels = torch_roi_size[0] * torch_roi_size[1]
    instance_loss_sum = torch.sum(instance_loss, dim=[1, 2])
    instance_loss_norm = torch.sum(instance_loss_sum / num_pixels)
    instance_loss_norm = instance_loss_norm * 0.25

    total_loss += instance_loss_norm
    writer.add_scalar('instance_loss_norm', instance_loss_norm, global_step)

    writer.add_scalar('total_loss', total_loss, global_step)

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    optimizer.step()

    return total_loss.data.item()


def test(sample_dict, epoch):
    model.eval()

    # Parse sample dict and remove incorrect batch of 1 (batched in kitti_dataset.py already)
    sample_name = sample_dict[constants.SAMPLE_NAME]
    num_objs = sample_dict[constants.SAMPLE_NUM_OBJS]
    i2_primes = sample_dict[constants.SAMPLE_I2_PRIME].squeeze()
    coord_u_grids = sample_dict[constants.SAMPLE_COORD_GRID_U].squeeze()

    img_2_input = sample_dict[constants.SAMPLE_IMAGE_2_INPUT].squeeze()
    img_3_input = sample_dict[constants.SAMPLE_IMAGE_3_INPUT].squeeze()
    full_img_2_input = sample_dict[constants.SAMPLE_FULL_IMAGE_2_INPUT]
    full_img_3_input = sample_dict[constants.SAMPLE_FULL_IMAGE_3_INPUT]

    rois_left = sample_dict[constants.SAMPLE_LABEL_ROIS_LEFT].squeeze()
    rois_right = sample_dict[constants.SAMPLE_LABEL_ROIS_RIGHT].squeeze()

    boxes_2d_left = sample_dict[constants.SAMPLE_LABEL_BOXES_2D_LEFT].squeeze()
    boxes_2d_right = sample_dict[constants.SAMPLE_LABEL_BOXES_2D_RIGHT].squeeze()

    # gt_disp_maps_local = sample_dict[constants.SAMPLE_DISP_MAPS_LOCAL].squeeze()
    disp_map = sample_dict[constants.SAMPLE_DISP_MAPS_GLOBAL].squeeze()
    gt_depth_maps = sample_dict[constants.SAMPLE_DEPTH_MAP_CROPS].squeeze()

    instance_masks = sample_dict[constants.SAMPLE_INSTANCE_MASKS].squeeze()

    stereo_calib_f = sample_dict[constants.SAMPLE_STEREO_CALIB_F]
    stereo_calib_b = sample_dict[constants.SAMPLE_STEREO_CALIB_B]
    stereo_calib_center_u = sample_dict[constants.SAMPLE_STEREO_CALIB_CENTER_U]
    stereo_calib_center_v = sample_dict[constants.SAMPLE_STEREO_CALIB_CENTER_V]

    # Set as Variables
    imgL = Variable(torch.FloatTensor(img_2_input))
    imgR = Variable(torch.FloatTensor(img_3_input))

    full_imgL = Variable(torch.FloatTensor(full_img_2_input))
    full_imgR = Variable(torch.FloatTensor(full_img_3_input))

    rois_left = Variable(torch.FloatTensor(rois_left))
    rois_right = Variable(torch.FloatTensor(rois_right))

    boxes_2d_left = Variable(torch.FloatTensor(boxes_2d_left))
    boxes_2d_right = Variable(torch.FloatTensor(boxes_2d_right))

    disp_l = Variable(torch.FloatTensor(disp_map.float()))
    gt_depth_maps = Variable(torch.FloatTensor(gt_depth_maps.float()))

    stereo_calib_f = Variable(torch.FloatTensor(stereo_calib_f), requires_grad=False)
    stereo_calib_b = Variable(torch.FloatTensor(stereo_calib_b), requires_grad=False)
    stereo_calib_center_u = Variable(torch.FloatTensor(stereo_calib_center_u), requires_grad=False)
    stereo_calib_center_v = Variable(torch.FloatTensor(stereo_calib_center_v), requires_grad=False)

    if args.cuda:
        imgL, imgR, full_imgL, full_imgR, disp_l, gt_depth_maps, \
            boxes_2d_left, boxes_2d_right, rois_left, rois_right, \
            stereo_calib_f, stereo_calib_b, stereo_calib_center_u, stereo_calib_center_v \
            = imgL.cuda(),\
            imgR.cuda(),\
            full_imgL.cuda(),\
            full_imgR,\
            disp_l.cuda(), \
            gt_depth_maps.cuda(), \
            boxes_2d_left.cuda(), \
            boxes_2d_right.cuda(), \
            rois_left.cuda(), \
            rois_right.cuda(), \
            stereo_calib_f.cuda(), stereo_calib_b.cuda(), \
            stereo_calib_center_u.cuda(), stereo_calib_center_v.cuda()\


    with torch.no_grad():
        # TODO: Check if this is valid
        # Have the same image twice for data parallelism
        full_imgL = torch.cat([full_imgL, full_imgL], 0)
        full_imgR = torch.cat([full_imgR, full_imgR], 0)

        output3, pred_instance_masks = model(
            imgL, imgR, full_imgL, full_imgR,
            rois_left, rois_right, kitti_train_dataset.full_img_downsample_scale)

    stereo_calib_f = stereo_calib_f.cpu().numpy()
    stereo_calib_b = stereo_calib_b.cpu().numpy()
    coord_u_grids = coord_u_grids.data.cpu().numpy()
    i2_primes = i2_primes.data.cpu().numpy()
    boxes_2d_right = boxes_2d_right.data.cpu().numpy()
    num_objs = num_objs.data.cpu()
    pred_disps_local = output3.data.cpu().numpy()
    disp_true = disp_l.data.cpu().numpy()
    instance_masks = instance_masks.data.cpu().numpy()
    pred_instance_masks = pred_instance_masks.data.cpu().numpy()
    gt_depth_maps = gt_depth_maps.cpu().numpy()

    # Save predicted instance masks
    pred_instance_masks = (pred_instance_masks > 0.0).astype(np.float32)
    pred_instance_masks = pred_instance_masks.astype(np.uint8) * 255
    mask_output_dir = os.path.join(masks_dir, str(epoch))
    if os.path.isdir(mask_output_dir):
        pass
    else:
        os.makedirs(mask_output_dir)
    for mask_idx, mask in enumerate(pred_instance_masks):
        cv2.imwrite(mask_output_dir + '/{}_{}.png'.format(sample_name[0], mask_idx), mask)

    # Remove scaling
    pred_disps_local = pred_disps_local / disp_scale_factor

    pred_global_disp = [instance_utils.calc_global_from_local_disp(
        pred_disp_local, coord_u_grid, i2_prime, box_2d_right, roi_size, instance_mask)
        for (box_2d_right, pred_disp_local, coord_u_grid, i2_prime, instance_mask)
        in zip(boxes_2d_right, pred_disps_local, coord_u_grids, i2_primes, instance_masks)]

    pred_global_disp = np.stack(pred_global_disp, axis=0)

    # Only compute error on the actual number of objects
    pred_global_disp = pred_global_disp[:num_objs, :, :]
    disp_true = disp_true[:num_objs, :, :]
    gt_depth_maps = gt_depth_maps[:num_objs, :, :]
    instance_masks = instance_masks[:num_objs, :, :]

    # Compute 3-px error
    true_disp = disp_true
    true_disp *= instance_masks

    if np.max(true_disp) == 0:
        torch.cuda.empty_cache()
        return None, None, None
    else:
        index = np.argwhere(true_disp != 0).T
        error_map = np.zeros_like(disp_true)

        error_map[index[0][:], index[1][:], index[2][:]] = \
            np.abs(true_disp[index[0][:], index[1][:], index[2][:]]
                   - pred_global_disp[index[0][:], index[1][:], index[2][:]])

        correct_3px = \
            (error_map[index[0][:], index[1][:], index[2][:]] < 3) | \
            (error_map[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:],
                                                                          index[1][:],
                                                                          index[2][:]] * 0.05)

        correct_1px = \
            (error_map[index[0][:], index[1][:], index[2][:]] < 1) | \
            (error_map[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:],
                                                                          index[1][:],
                                                                          index[2][:]] * 0.05)

        error_3px = 1 - (float(sum(correct_3px)) / float(len(index[0])))
        error_1px = 1 - (float(sum(correct_1px)) / float(len(index[0])))

        # # # Calculate depth errors # # #
        gt_depth_maps *= instance_masks

        pred_global_disp[pred_global_disp <= 0] = 0.1
        pred_depths = (stereo_calib_f * stereo_calib_b) / pred_global_disp
        # Discard pixels past 80m
        pred_depths[pred_depths > 80.0] = 0.0

        # RMSE
        depth_index = np.argwhere(gt_depth_maps > 0.0).T
        diff = np.abs(gt_depth_maps[depth_index[0][:], depth_index[1][:], depth_index[2][:]]
                      - pred_depths[depth_index[0][:], depth_index[1][:], depth_index[2][:]])
        diff_squared = diff ** 2
        rmse = np.sqrt(diff_squared.mean())

        torch.cuda.empty_cache()

        return error_3px, error_1px, rmse


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 8:
        lr = 0.001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    max_acc = 0
    max_epo = 0
    global_step = 0
    start_full_time = time.time()

    error_path = save_dir + '/error.csv'

    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0
        total_test_3_px_loss = 0
        total_test_1_px_loss = 0
        total_rmse = 0
        adjust_learning_rate(optimizer, epoch)

        # Training
        num_valid_train_samples = 0
        for batch_idx, sample_dict in enumerate(TrainImgLoader):
            start_time = time.time()

            if not sample_dict:
                continue
            else:
                num_valid_train_samples += 1

            # Compute loss
            loss = train(sample_dict, global_step)

            print('Iter %d training loss = %.3f , time = %.2f' %
                  (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
            global_step += 1

        print('epoch %d total training loss = %.3f' %
              (epoch, total_train_loss / num_valid_train_samples))

        # Validation
        num_valid_val_samples = 0
        for batch_idx, sample_dict in enumerate(TestImgLoader):

            if not sample_dict:
                continue

            # Calculate 3 pixel error
            test_loss_3px, test_loss_1px, rmse = test(sample_dict, epoch)

            if test_loss_3px is None or test_loss_1px is None:
                pass
            else:
                num_valid_val_samples += 1
                print('Iter %d / %d val 3-px error = %.3f' % (batch_idx, len(TestImgLoader),
                                                              test_loss_3px * 100))
                total_test_3_px_loss += test_loss_3px
                total_test_1_px_loss += test_loss_1px
                total_rmse += rmse

        print('epoch %d total 3-px error in val = %.3f' %
              (epoch, total_test_3_px_loss / num_valid_val_samples * 100))
        if total_test_3_px_loss / num_valid_val_samples * 100 > max_acc:
            max_acc = total_test_3_px_loss / num_valid_val_samples * 100
            max_epo = epoch
        print('MAX epoch %d total test error = %.3f' % (max_epo, max_acc))

        # Save 3 px error
        error_file = open(error_path, 'a')
        csv_writer = csv.writer(error_file, delimiter=',')
        three_pixel_error = total_test_3_px_loss / num_valid_val_samples * 100.0
        one_pixel_error = total_test_1_px_loss / num_valid_val_samples * 100.0
        rmse_error = total_rmse / num_valid_val_samples * 100.0
        csv_writer.writerow([three_pixel_error, one_pixel_error, rmse_error])
        error_file.close()

        # SAVE
        savefilename = save_dir + '/finetune_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': total_train_loss / num_valid_train_samples,
            'test_loss': total_test_3_px_loss / num_valid_val_samples,
        }, savefilename)

    print('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    print(max_epo)
    print(max_acc)


if __name__ == '__main__':
    main()
