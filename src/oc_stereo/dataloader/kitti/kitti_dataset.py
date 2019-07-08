import fnmatch
import os
from PIL import Image

import cv2
import numpy as np
import torch.utils.data as data

import oc_stereo
from oc_stereo.core import constants, box_3d_projector
from oc_stereo.dataloader import preprocess
from oc_stereo.dataloader.kitti import calib_utils, kitti_aug, depth_map_utils
from oc_stereo.dataloader.kitti import instance_utils
from oc_stereo.dataloader.kitti import obj_utils
from oc_stereo.dataloader.kitti.obj_utils import Difficulty


class Sample:
    def __init__(self, name, augs):
        self.name = name
        self.augs = augs

    def __repr__(self):
        return '({}, augs: {})'.format(self.name, self.augs)


class KittiDataset(data.Dataset):

    def __init__(self, dataset_config, train_val_test):

        self.dataset_config = dataset_config
        self.train_val_test = train_val_test

        # Parse config
        self.name = self.dataset_config.name

        self.data_split = self.dataset_config.data_split
        self.dataset_dir = os.path.expanduser(self.dataset_config.dataset_dir)
        data_split_dir = self.dataset_config.data_split_dir

        self.num_boxes = self.dataset_config.num_boxes

        self.classes = list(self.dataset_config.classes)
        if self.classes == ['Car']:
            self.obj_type = 'car'
        elif self.classes == ['Pedestrian', 'Cyclist']:
            self.obj_type = 'ped_cyc'
        else:
            raise ValueError('Invalid class', self.classes)

        self.num_classes = len(self.classes)

        # Object filtering config
        if self.train_val_test in ['train', 'val']:
            obj_filter_config = self.dataset_config.obj_filter_config
            obj_filter_config.classes = self.classes
            self.obj_filter = obj_utils.ObjectFilter(obj_filter_config)

        elif self.train_val_test == 'test':
            # Use all detections during inference
            self.obj_filter = obj_utils.ObjectFilter.create_obj_filter(
                classes=self.classes,
                difficulty=Difficulty.ALL,
                occlusion=None,
                truncation=None,
                box_2d_height=None,
                depth_range=None)

        else:
            raise ValueError('Invalid train_val_test', train_val_test)

        self.car_obj_filter = obj_utils.ObjectFilter.create_obj_filter(
            classes=['Car'],
            difficulty=obj_utils.Difficulty.ALL,
            occlusion=None,
            truncation=None,
            box_2d_height=None,
            depth_range=None)

        self.has_kitti_labels = self.dataset_config.has_kitti_labels

        # Detection settings
        self.det_type = self.dataset_config.det_type
        self.use_mscnn_detections = self.dataset_config.use_mscnn_detections
        self.det_thr = self.dataset_config.det_thr
        self.mscnn_instance_version = self.dataset_config.mscnn_instance_version
        self.use_srgt_masks = self.dataset_config.use_srgt_masks

        self.classes_name = self._set_up_classes_name()

        if self.classes_name == 'Car':
            self.mscnn_merge_min_iou = 0.7
        elif self.classes_name in ['Pedestrian', 'Cyclist']:
            self.mscnn_merge_min_iou = 0.5

        # Check that paths and split are valid
        self._check_dataset_dir()
        all_dataset_files = os.listdir(self.dataset_dir)
        self._check_data_split_valid(all_dataset_files)
        self.data_split_dir = self._check_data_split_dir_valid(
            all_dataset_files, data_split_dir)

        self.depth_version = self.dataset_config.depth_version
        self.disp_version = self.dataset_config.disp_version
        self.instance_version = self.dataset_config.instance_version

        # Setup directories
        self._set_up_directories()

        # Whether to oversample objects to required number of boxes
        self.oversample = self.dataset_config.oversample

        # Augmentation
        self.aug_config = self.dataset_config.aug_config
        self.jitter_iou = self.dataset_config.aug_config.jitter_iou

        # Initialize the sample list
        loaded_sample_names = self.load_sample_names(self.data_split)
        all_samples = [Sample(sample_name, []) for sample_name in loaded_sample_names]

        self.sample_list = np.asarray(all_samples)
        self.num_samples = len(self.sample_list)

        # Full image information
        self.full_image_crop_shape = self.dataset_config.full_img_crop_shape
        self.full_img_size = self.dataset_config.full_img_size
        self.full_img_downsample_scale = np.float32(self.full_image_crop_shape[1] /
                                                    self.full_img_size[0])

        # RoI Size
        self.roi_size = self.dataset_config.roi_size
        self.disp_scale_factor = self.dataset_config.disp_scale_factor
        self.disp_range = self.dataset_config.disp_range

    def _check_dataset_dir(self):
        """Checks that dataset directory exists in the file system

        Raises:
            FileNotFoundError: if the dataset folder is missing
        """
        # Check that dataset path is valid
        if not os.path.exists(self.dataset_dir):
            raise ValueError('Dataset path does not exist: {}'.format(self.dataset_dir))

    def _check_data_split_valid(self, all_dataset_files):
        possible_splits = []
        for file_name in all_dataset_files:
            if fnmatch.fnmatch(file_name, '*.txt'):
                possible_splits.append(os.path.splitext(file_name)[0])
        # This directory contains a readme.txt file, remove it from the list
        if 'readme' in possible_splits:
            possible_splits.remove('readme')

        if self.data_split not in possible_splits:
            raise ValueError("Invalid data split: {}, possible_splits: {}"
                             .format(self.data_split, possible_splits))

    def _check_data_split_dir_valid(self, all_dataset_files, data_split_dir):
        # Check data_split_dir
        # Get possible data split dirs from folder names in dataset folder
        possible_split_dirs = []
        for folder_name in all_dataset_files:
            if os.path.isdir(self.dataset_dir + '/' + folder_name):
                possible_split_dirs.append(folder_name)

        if data_split_dir in possible_split_dirs:
            # Overwrite with full path
            data_split_dir = self.dataset_dir + '/' + data_split_dir
        else:
            raise ValueError(
                "Invalid data split dir: {}, possible dirs".format(
                    data_split_dir, possible_split_dirs))

        return data_split_dir

    def _set_up_directories(self):
        """Sets up data directories."""
        # Setup Directories
        self.image_2_dir = self.data_split_dir + '/image_2'
        self.image_3_dir = self.data_split_dir + '/image_3'

        self.calib_dir = self.data_split_dir + '/calib'
        self.planes_dir = self.data_split_dir + '/planes'
        self.velo_dir = self.data_split_dir + '/velodyne'
        self.disp_dir = self.data_split_dir + '/disparity_{}'.format(self.disp_version)
        self.coarse_disp_dir = oc_stereo.data_dir() + '/coarse_disp/disparity_psmnet'
        self.depth_2_dir = self.data_split_dir + '/depth_{}_{}'.format(
            2, self.depth_version)
        self.instance_2_dir = self.data_split_dir + '/instance_{}_{}'.format(
            2, self.instance_version)
        self.srgt_instance_2_dir = self.data_split_dir + '/instance_{}_{}'.format(
            2, 'srgt')

        self.det_label_2_dir = oc_stereo.data_dir() + \
            '/detections/{}/kitti_fmt/{}_matching_{}/{}_2_matching_{}/data'.format(
                self.det_type, self.data_split.split('_')[0],
                '_'.join(map(str, self.det_thr)), self.obj_type,
                '_'.join(map(str, self.det_thr)))
        self.det_label_3_dir = oc_stereo.data_dir() + \
            '/detections/{}/kitti_fmt/{}_matching_{}/{}_3_matching_{}/data'.format(
                self.det_type, self.data_split.split('_')[0],
                '_'.join(map(str, self.det_thr)), self.obj_type,
                '_'.join(map(str, self.det_thr)))

        self.mscnn_instance_2_dir = self.data_split_dir + '/instance_{}_{}'.format(
            2, self.mscnn_instance_version)

        if self.has_kitti_labels:
            self.kitti_label_dir = self.data_split_dir + '/label_2'

    def _set_up_classes_name(self):
        # Unique identifier for multiple classes
        if self.num_classes > 1:
            if self.classes == ['Pedestrian', 'Cyclist']:
                classes_name = 'People'
            elif self.classes == ['Car', 'Pedestrian', 'Cyclist']:
                classes_name = 'All'
        else:
            classes_name = self.classes[0]

        return classes_name

    def get_sample_names(self):
        return [sample.name for sample in self.sample_list]

    # Get sample paths
    def get_image_2_path(self, sample_name):
        return self.image_2_dir + '/' + sample_name + '.png'

    def get_image_3_path(self, sample_name):
        return self.image_3_dir + '/' + sample_name + '.png'

    def get_depth_map_path(self, sample_name):
        return self.depth_2_dir + '/' + sample_name + '_left_depth.png'

    def get_velodyne_path(self, sample_name):
        return self.velo_dir + '/' + sample_name + '.bin'

    def load_sample_names(self, data_split):
        """Load the sample names listed in this dataset's set file
        (e.g. train.txt, validation.txt)

        Args:
            data_split: the sample list to load

        Returns:
            A list of sample names (file names) read from
            the .txt file corresponding to the data split
        """
        set_file = self.dataset_dir + '/' + data_split + '.txt'
        with open(set_file, 'r') as f:
            sample_names = f.read().splitlines()

        return np.asarray(sample_names)

    def __getitem__(self, index):

        sample_idx = index

        sample = self.sample_list[sample_idx]
        sample_name = sample.name

        # Load image
        image_2_input = np.asarray(
            Image.open(self.get_image_2_path(sample_name)).convert('RGB'))
        image_3_input = np.asarray(
            Image.open(self.get_image_3_path(sample_name)).convert('RGB'))

        image_2_shape = np.shape(image_2_input)[0:2]
        image_3_shape = np.shape(image_3_input)[0:2]

        # Crop full image
        image_2_input_cropped = image_2_input[
            -self.full_image_crop_shape[0]:, -self.full_image_crop_shape[1]:, :]
        image_3_input_cropped = image_3_input[
            -self.full_image_crop_shape[0]:, -self.full_image_crop_shape[1]:, :]

        # Get calibration
        frame_calib = calib_utils.get_frame_calib(self.calib_dir, sample_name)
        stereo_calib = calib_utils.get_stereo_calibration(frame_calib.p2,
                                                          frame_calib.p3)
        cam_p_2 = frame_calib.p2
        cam_p_3 = frame_calib.p3

        # Return False is no valid boxes
        if self.train_val_test in ['train'] and not self.use_mscnn_detections:

            # Read KITTI object labels
            obj_labels = obj_utils.read_labels(self.kitti_label_dir, sample_name)
            num_all_objs = len(obj_labels)

            if num_all_objs < 1:
                return False

            if self.use_srgt_masks:
                # Count the number of cars
                car_obj_labels, car_obj_mask = obj_utils.apply_obj_filter(obj_labels,
                                                                          self.car_obj_filter)
                num_cars = len(car_obj_labels)
                if num_cars < 1:
                    return False

                # Filter labels further
                obj_labels, obj_mask = obj_utils.apply_obj_filter(car_obj_labels,
                                                                  self.obj_filter)
                num_objs = len(obj_labels)
                if num_objs < 1:
                    return False

                # Load SRGT instance masks
                srgt_instance_image = instance_utils.get_instance_image(sample_name,
                                                                        self.srgt_instance_2_dir)
                srgt_instance_masks = instance_utils.get_instance_mask_list(srgt_instance_image,
                                                                            num_cars)
                srgt_instance_masks = srgt_instance_masks[obj_mask]
                srgt_instance_masks = srgt_instance_masks.astype(np.int32)

                # Load regular instance masks
                instance_image = instance_utils.get_instance_image(sample_name,
                                                                   self.instance_2_dir)
                instance_masks = instance_utils.get_instance_mask_list(instance_image, num_all_objs)
                instance_masks = instance_masks[car_obj_mask]
                instance_masks = instance_masks[obj_mask]
                instance_masks = instance_masks.astype(np.int32)

            else:
                obj_labels, obj_mask = obj_utils.apply_obj_filter(obj_labels, self.obj_filter)

                # Load instance masks
                instance_image = instance_utils.get_instance_image(sample_name, self.instance_2_dir)
                instance_masks = instance_utils.get_instance_mask_list(instance_image, num_all_objs)
                instance_masks = instance_masks[obj_mask]
                instance_masks = instance_masks.astype(np.int32)

            # Load disparity maps
            gt_disp_map = obj_utils.get_disp_map(sample_name, self.disp_dir)
            gt_depth_map = obj_utils.get_depth_map(sample_name, self.depth_2_dir)

            # Get 3D boxes
            label_boxes_3d = obj_utils.boxes_3d_from_obj_labels(obj_labels)

            # Project 3D boxes to obtain 2D boxes:
            label_boxes_2d_left = []
            label_boxes_2d_right = []
            valid_instance_masks = []
            valid_srgt_instance_masks = []
            for idx, box_3d in enumerate(label_boxes_3d):

                box_2d_left = box_3d_projector.project_to_image_space(box_3d, cam_p_2,
                                                                      truncate=True,
                                                                      image_shape=image_2_shape)
                # Skip boxes that are too truncated
                if box_2d_left is None:
                    continue

                box_2d_right = box_3d_projector.project_to_image_space(box_3d, cam_p_3,
                                                                       truncate=True,
                                                                       image_shape=image_3_shape)
                if box_2d_right is None:
                    continue

                # Check if instance mask has points
                instance_mask = instance_masks[idx]
                if np.max(instance_mask) == 0:
                    continue

                # Jitter the 2D boxes
                if self.aug_config.use_box_jitter:
                    box_2d_left = kitti_aug.jitter_single_box_2d(box_2d_left,
                                                                 self.jitter_iou,
                                                                 image_2_shape)
                    box_2d_right = kitti_aug.jitter_single_box_2d(box_2d_right,
                                                                  self.jitter_iou,
                                                                  image_3_shape)

                if self.use_srgt_masks:
                    valid_srgt_instance_masks.append(srgt_instance_masks[idx])
                valid_instance_masks.append(instance_masks[idx])
                label_boxes_2d_left.append(box_2d_left)
                label_boxes_2d_right.append(box_2d_right)

            # Return false if no valid objects
            if len(valid_instance_masks) == 0:
                return False
            else:
                valid_instance_masks = np.stack(valid_instance_masks)
                if self.use_srgt_masks:
                    valid_srgt_instance_masks = np.stack(valid_srgt_instance_masks)

            # Make boxes equal in height
            boxes_2d_left_same_height, boxes_2d_right_same_height \
                = instance_utils.make_boxes_same_height(label_boxes_2d_left,
                                                        label_boxes_2d_right)

            inst_depth_crops, inst_valid_masks = instance_utils.np_instance_crop(
                boxes_2d=boxes_2d_left_same_height,
                boxes_3d=boxes_2d_right_same_height,
                instance_masks=instance_masks,
                input_map=np.expand_dims(gt_depth_map, 2),
                roi_size=self.roi_size,
                view_norm=False)

            camN_inst_pc_maps = [depth_map_utils.depth_patch_to_pc_map(
                inst_depth_crop, box_2d, cam_p_2, self.roi_size,
                depth_map_shape=gt_depth_map.shape[0:2],
                use_pixel_centres=False, use_corr_factors=False)
                for inst_depth_crop, box_2d in zip(inst_depth_crops, boxes_2d_left_same_height)]

        elif self.train_val_test in ['val', 'test'] or self.use_mscnn_detections:

            # Read object labels
            obj_labels_2 = obj_utils.read_labels(self.det_label_2_dir, sample_name)
            obj_labels_3 = obj_utils.read_labels(self.det_label_3_dir, sample_name)

            num_all_objs = len(obj_labels_2)
            if num_all_objs < 1:
                return False

            if self.train_val_test in ['train', 'val']:
                # Load disparity maps
                gt_disp_map = obj_utils.get_disp_map(sample_name, self.disp_dir)
                gt_depth_map = obj_utils.get_depth_map(sample_name, self.depth_2_dir)

                # Load in instance masks
                instance_image = instance_utils.get_instance_image(sample_name,
                                                                   self.mscnn_instance_2_dir)
                instance_masks = instance_utils.get_instance_mask_list(instance_image,
                                                                       num_all_objs)
                valid_instance_masks = instance_masks.astype(np.int32)

            # Get 2D boxes
            label_boxes_2d_left = obj_utils.boxes_2d_from_obj_labels(obj_labels_2)
            label_boxes_2d_right = obj_utils.boxes_2d_from_obj_labels(obj_labels_3)

            # Make boxes equal in height
            boxes_2d_left_same_height, boxes_2d_right_same_height \
                = instance_utils.make_boxes_same_height(label_boxes_2d_left,
                                                        label_boxes_2d_right)

        else:
            raise ValueError('Invalid run mode', self.train_val_test)

        # Preprocessor
        processed = preprocess.get_transform(augment=False)

        # Save unscaled boxes
        box_2d_left_same_height_unscaled = boxes_2d_left_same_height
        box_2d_right_same_height_unscaled = boxes_2d_right_same_height

        # Get cropped inputs and ground truth
        all_i2_prime = []
        all_coord_u = []
        all_img_2_crops = []
        all_img_3_crops = []
        all_mask_crops = []
        all_srgt_mask_crops = []
        all_local_gt_disp_maps = []
        all_global_gt_disp_maps = []
        all_gt_depth_maps = []
        all_global_xyz_maps = []
        all_inst_xyz_masks = []
        for idx, (box_left, box_right, box_left_unscaled, box_right_unscaled) in enumerate(
                zip(boxes_2d_left_same_height, boxes_2d_right_same_height,
                    box_2d_left_same_height_unscaled, box_2d_right_same_height_unscaled)):

            # Get 2D box left coordinate points within roi_size grid
            coord_grid = instance_utils.get_exp_proj_uv_map(box_left, self.roi_size)
            coord_grid_u = np.array(coord_grid[:, :, 0])

            i2_prime = instance_utils.calc_i2_prime(self.roi_size)

            if self.train_val_test in ['train', 'val']:

                # Instance mask crop and resize
                instance_mask = valid_instance_masks[idx]
                instance_mask_crop = instance_utils.get_valid_inst_box_2d_crop(box_left_unscaled,
                                                                               instance_mask)
                instance_mask_crop_resized = cv2.resize(instance_mask_crop, (self.roi_size[1],
                                                                             self.roi_size[0]),
                                                        interpolation=cv2.INTER_NEAREST)

                if self.use_srgt_masks and self.train_val_test == 'train':
                    srgt_instance_mask = valid_srgt_instance_masks[idx]

                    # Instance srgt mask crop and resize
                    srgt_instance_mask_crop = instance_utils.get_valid_inst_box_2d_crop(
                        box_left_unscaled, srgt_instance_mask)
                    srgt_instance_mask_crop_resized = cv2.resize(
                        srgt_instance_mask_crop, (self.roi_size[1], self.roi_size[0]), interpolation=cv2.INTER_NEAREST)
                    all_srgt_mask_crops.append(srgt_instance_mask_crop_resized)

                # Mask ground truth disparity and depth map
                gt_disp_map_masked = gt_disp_map * instance_mask
                gt_depth_map_masked = gt_depth_map * instance_mask

                # Get gt local disp map
                local_gt_disp_map = instance_utils.calc_local_disp(coord_grid_u, i2_prime, box_left,
                                                                   box_right, gt_disp_map_masked,
                                                                   self.roi_size,
                                                                   instance_mask_crop_resized)
                # Calculate mask
                mask = (local_gt_disp_map > self.disp_range[0]) & \
                       (local_gt_disp_map < self.disp_range[1]) & \
                       (local_gt_disp_map != 0)

                if len(local_gt_disp_map[mask]) == 0:
                    continue

                # Scale local disparity map
                scaled_local_gt_disp_map = self.disp_scale_factor * local_gt_disp_map

                # Get global disp crop
                gt_disp_map_masked_crop = instance_utils.get_valid_inst_box_2d_crop(
                    box_left, gt_disp_map_masked)
                global_gt_disp_map = cv2.resize(gt_disp_map_masked_crop,
                                                (self.roi_size[1], self.roi_size[0]),
                                                interpolation=cv2.INTER_NEAREST)
                gt_depth_map_masked_crop = instance_utils.get_valid_inst_box_2d_crop(
                    box_left, gt_depth_map_masked)
                gt_depth_map_masked_crop_resized = cv2.resize(gt_depth_map_masked_crop,
                                                              (self.roi_size[1], self.roi_size[0]),
                                                              interpolation=cv2.INTER_NEAREST)
                all_local_gt_disp_maps.append(scaled_local_gt_disp_map)
                all_global_gt_disp_maps.append(global_gt_disp_map)
                all_gt_depth_maps.append(gt_depth_map_masked_crop_resized)
                all_mask_crops.append(instance_mask_crop_resized)
                if self.train_val_test == 'train':
                    all_global_xyz_maps.append(camN_inst_pc_maps[idx])
                    all_inst_xyz_masks.append(inst_valid_masks[idx])

            # Get cropped images
            img_2_crop = instance_utils.get_valid_inst_box_2d_crop(box_left,
                                                                   image_2_input)
            img_3_crop = instance_utils.get_valid_inst_box_2d_crop(box_right,
                                                                   image_3_input)

            # Resize
            img_2_crop_resized = cv2.resize(img_2_crop, (self.roi_size[1], self.roi_size[0]),
                                            interpolation=cv2.INTER_LINEAR)
            img_3_crop_resized = cv2.resize(img_3_crop, (self.roi_size[1], self.roi_size[0]),
                                            interpolation=cv2.INTER_LINEAR)

            img_2_preprocessed = processed(img_2_crop_resized)
            img_3_preprocessed = processed(img_3_crop_resized)
            all_coord_u.append(coord_grid_u)
            all_i2_prime.append(i2_prime)
            all_img_2_crops.append(img_2_preprocessed)
            all_img_3_crops.append(img_3_preprocessed)

        # If no valid objects return False
        if len(all_img_2_crops) == 0:
            return False

        # Stack all crops
        batched_i2_prime = np.stack(all_i2_prime)
        batched_coord_u = np.stack(all_coord_u)
        batched_img_2_crops = np.stack(all_img_2_crops)
        batched_img_3_crops = np.stack(all_img_3_crops)
        num_valid_objs = np.shape(batched_img_2_crops)[0]

        if self.train_val_test in ['train', 'val']:
            batched_mask_crops = np.stack(all_mask_crops)
            batched_local_gt_disp_maps = np.stack(all_local_gt_disp_maps)
            batched_global_gt_disp_maps = np.stack(all_global_gt_disp_maps)
            batched_gt_depth_maps = np.stack(all_gt_depth_maps)

            if self.use_srgt_masks and self.train_val_test == 'train':
                batched_srgt_mask_crops = np.stack(all_srgt_mask_crops)

            # Oversample to required number of boxes to keep consistent batch size
            if self.oversample:
                num_to_oversample = self.num_boxes - num_valid_objs

                oversample_indices = np.random.choice(
                    num_valid_objs, num_to_oversample, replace=True)
                oversample_indices = np.hstack([np.arange(0, num_valid_objs), oversample_indices])
                batched_i2_prime = batched_i2_prime[oversample_indices]
                batched_coord_u = batched_coord_u[oversample_indices]
                batched_img_2_crops = batched_img_2_crops[oversample_indices]
                batched_img_3_crops = batched_img_3_crops[oversample_indices]
                boxes_2d_left_same_height = boxes_2d_left_same_height[oversample_indices]
                boxes_2d_right_same_height = boxes_2d_right_same_height[oversample_indices]

                if self.train_val_test in ['train', 'val']:
                    batched_mask_crops = batched_mask_crops[oversample_indices]
                    batched_local_gt_disp_maps = batched_local_gt_disp_maps[oversample_indices]
                    batched_global_gt_disp_maps = batched_global_gt_disp_maps[oversample_indices]
                    batched_gt_depth_maps = batched_gt_depth_maps[oversample_indices]

                if self.train_val_test == 'train':

                    if self.use_srgt_masks:
                        batched_srgt_mask_crops = batched_srgt_mask_crops[oversample_indices]
                    batched_global_xyz_maps = np.stack(all_global_xyz_maps)
                    batched_inst_xyz_masks = np.stack(all_inst_xyz_masks)
                    batched_global_xyz_maps = batched_global_xyz_maps[oversample_indices]
                    batched_inst_xyz_masks = batched_inst_xyz_masks[oversample_indices]

        # Process the full left and right images
        full_img2_resized = cv2.resize(image_2_input_cropped, tuple(self.full_img_size),
                                       interpolation=cv2.INTER_LINEAR)
        full_img3_resized = cv2.resize(image_3_input_cropped, tuple(self.full_img_size),
                                       interpolation=cv2.INTER_LINEAR)
        full_img_2 = processed(full_img2_resized)
        full_img_3 = processed(full_img3_resized)

        # Create RoIs (adjust boxes to cropping) by shifting x and y coordinates
        left_rois = np.copy(boxes_2d_left_same_height)
        right_rois = np.copy(boxes_2d_right_same_height)

        shape_diff = np.asarray(image_2_shape) - self.full_image_crop_shape
        x_diff = shape_diff[1]
        y_diff = shape_diff[0]

        # Shift x coordinates
        left_rois[:, 1] = np.maximum(left_rois[:, 1] - x_diff, 0)
        left_rois[:, 3] = np.maximum(left_rois[:, 3] - x_diff, 0)
        right_rois[:, 1] = np.maximum(right_rois[:, 1] - x_diff, 0)
        right_rois[:, 3] = np.maximum(right_rois[:, 3] - x_diff, 0)

        # Shift y coordinates
        left_rois[:, 0] = np.maximum(left_rois[:, 0] - y_diff, 0)
        left_rois[:, 2] = np.maximum(left_rois[:, 2] - y_diff, 0)
        right_rois[:, 0] = np.maximum(right_rois[:, 0] - y_diff, 0)
        right_rois[:, 2] = np.maximum(right_rois[:, 2] - y_diff, 0)

        # Change RoIs to [x1, y1, x2, y2] order
        left_rois = left_rois[:, [1, 0, 3, 2]]
        right_rois = right_rois[:, [1, 0, 3, 2]]

        if self.train_val_test in ['train', 'val']:
            # Add image ID to RoIs (all image 0)
            left_rois = np.concatenate([np.zeros((self.num_boxes, 1)), left_rois], axis=1)
            right_rois = np.concatenate([np.zeros((self.num_boxes, 1)), right_rois], axis=1)
        else:
            # Add image ID to RoIs (all image 0)
            left_rois = np.concatenate([np.zeros((num_valid_objs, 1)), left_rois], axis=1)
            right_rois = np.concatenate([np.zeros((num_valid_objs, 1)), right_rois], axis=1)

        sample_dict = {
            constants.SAMPLE_NUM_OBJS: num_valid_objs,

            constants.SAMPLE_COORD_GRID_U: batched_coord_u,
            constants.SAMPLE_I2_PRIME: batched_i2_prime,

            constants.SAMPLE_IMAGE_2_SHAPE: image_2_shape,
            constants.SAMPLE_IMAGE_2_INPUT: batched_img_2_crops,
            constants.SAMPLE_IMAGE_3_INPUT: batched_img_3_crops,
            constants.SAMPLE_FULL_IMAGE_2_INPUT: full_img_2,
            constants.SAMPLE_FULL_IMAGE_3_INPUT: full_img_3,

            constants.SAMPLE_LABEL_ROIS_LEFT: np.float32(left_rois),
            constants.SAMPLE_LABEL_ROIS_RIGHT: np.float32(right_rois),
            constants.SAMPLE_LABEL_BOXES_2D_LEFT: np.float32(boxes_2d_left_same_height),
            constants.SAMPLE_LABEL_BOXES_2D_RIGHT: np.float32(boxes_2d_right_same_height),

            constants.SAMPLE_CAM_P: cam_p_2,
            constants.SAMPLE_STEREO_CALIB_F: np.float32(stereo_calib.f),
            constants.SAMPLE_STEREO_CALIB_B: np.float32(stereo_calib.baseline),
            constants.SAMPLE_STEREO_CALIB_CENTER_U: np.float32(stereo_calib.center_u),
            constants.SAMPLE_STEREO_CALIB_CENTER_V: np.float32(stereo_calib.center_v),
            constants.SAMPLE_NAME: sample_name,

        }

        if self.train_val_test == 'train':

            if self.use_srgt_masks:

                sample_dict.update({
                    constants.SAMPLE_SRGT_INSTANCE_MASKS: batched_srgt_mask_crops,
                })

            sample_dict.update({
                constants.SAMPLE_GT_GLOBAL_XYZ_MAP: batched_global_xyz_maps,
                constants.SAMPLE_INST_XYZ_MASKS: np.float32(batched_inst_xyz_masks),
            })

        if self.train_val_test in ['train', 'val']:

            sample_dict.update({
                constants.SAMPLE_INSTANCE_MASKS: batched_mask_crops,
                constants.SAMPLE_DISP_MAPS_LOCAL: batched_local_gt_disp_maps,
                constants.SAMPLE_DISP_MAPS_GLOBAL: batched_global_gt_disp_maps,
                constants.SAMPLE_DEPTH_MAP_CROPS: batched_gt_depth_maps,

            })

        elif self.train_val_test == 'test':
            # No additional labels for test mode
            pass

        return sample_dict

    def __len__(self):
        return self.num_samples
