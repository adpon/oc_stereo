import os
import sys

import cv2
import numpy as np

import oc_stereo
from oc_stereo.core import config_utils
from oc_stereo.dataloader.kitti import instance_utils, obj_utils, evaluation
from oc_stereo.dataloader.kitti.kitti_dataset import KittiDataset
from oc_stereo.dataloader.kitti.obj_utils import ObjectFilter


def main():

    ##################
    # Options
    ##################
    data_split = 'train_matching_0.5'
    # data_split = 'val_matching_0.5'

    mscnn_thresh = [0.5]
    # mscnn_thresh = [0.9, 0.9]
    mask_type = 'multiscale'

    obj_type = 'car'
    # obj_type = 'ped_cyc'

    # Load config and overwrite data split
    default_config_path = oc_stereo.root_dir() + '/configs/default_train_config.yaml'
    config = config_utils.parse_yaml_config(default_config_path)
    dataset_config = config.dataset_config
    dataset_config.data_split = data_split

    # Create dataset
    dataset = KittiDataset(dataset_config, train_val_test='train')

    instance_name = 'trainval'
    mscnn_label_2_dir = oc_stereo.data_dir() + \
        '/detections/mscnn/kitti_fmt/{}/{}_2_matching_{}/data'.format(data_split,
                                                                      obj_type,
                                                                      '_'.join(map(str, mscnn_thresh)))

    create_new_data_split = True
    ##################

    # Object filter for cars
    obj_filter = ObjectFilter.create_obj_filter(
        classes=['Car'],
        difficulty=obj_utils.Difficulty.ALL,
        occlusion=None,
        truncation=None,
        box_2d_height=None,
        depth_range=None)

    # Make output folder
    output_dir = 'outputs/instance_2_{}'.format(instance_name)

    if os.path.isdir(output_dir):
        pass
    else:
        os.makedirs(output_dir)

    num_samples = dataset.num_samples
    sample_names = []

    if mask_type == 'multiscale':
        instance_dir = os.path.expanduser('~/Kitti/object/training/instance_2_depth_2_multiscale')
    else:
        raise ValueError('Invalid mask type', mask_type)

    for sample_idx in range(num_samples):

        sample_name = dataset.sample_list[sample_idx].name
        sys.stdout.write('\r{} / {}'.format(sample_idx + 1, num_samples))

        # Load MSCNN labels
        mscnn_obj_labels = obj_utils.read_labels(mscnn_label_2_dir, sample_name)
        mscnn_boxes = obj_utils.boxes_2d_from_obj_labels(mscnn_obj_labels)

        # Load KITTI labels
        kitti_obj_labels = obj_utils.read_labels(dataset.kitti_label_dir, sample_name)
        num_all_objs = len(kitti_obj_labels)

        if obj_type == 'car':
            # Filter to ensure only matches with the same class
            kitti_obj_labels, obj_mask = obj_utils.apply_obj_filter(kitti_obj_labels, obj_filter)

            num_cars = len(kitti_obj_labels)

            if num_cars == 0:
                continue

        kitti_boxes = obj_utils.boxes_2d_from_obj_labels(kitti_obj_labels)

        # Load in instance masks
        instance_image = instance_utils.get_instance_image(sample_name, instance_dir)

        if mask_type == 'multiscale':
            instance_masks = instance_utils.get_instance_mask_list(instance_image,
                                                                   num_all_objs)

        # Create blank instance image information
        blank_instance_image = np.full(np.shape(instance_image), 255, dtype=np.uint8)

        # Start instance index at 0 and generate instance masks for all boxes
        inst_idx = 0
        no_mask_count = 0
        used_masks = []
        # Find the KITTI box that has the greatest IoU to find appropriate instance mask
        for mscnn_idx, mscnn_box in enumerate(mscnn_boxes):
            iou_list = evaluation.two_d_iou(mscnn_box, kitti_boxes)
            matching_box_idx = np.argmax(iou_list)
            matching_iou = iou_list[matching_box_idx]

            if matching_box_idx in used_masks:
                inst_idx += 1
                continue

            if matching_iou > 0.6:
                instance_mask = instance_masks[matching_box_idx]
                blank_instance_image[instance_mask] = np.uint8(inst_idx)
                used_masks.append(matching_box_idx)
            else:
                # No valid instance mask
                no_mask_count += 1
                pass

            inst_idx += 1

        if no_mask_count == len(mscnn_boxes):
            print(' No valid masks')
            continue

        # Save new instance image
        cv2.imwrite(output_dir + '/{}.png'.format(sample_name), blank_instance_image,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])

        sample_names.append(sample_name)

    # Create split file
    if create_new_data_split:
        with open(os.path.expanduser(
                '~/Kitti/object/' + data_split + '_' + instance_name + '.txt'), 'w') as f:
            for sample_name in sample_names:
                f.write("%s\n" % sample_name)


if __name__ == '__main__':
    main()
