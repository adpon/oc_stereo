import copy
import os
import sys
import time

import cv2
import numpy as np
from skimage import measure

import oc_stereo
from oc_stereo.builders.dataset_builder import DatasetBuilder
from oc_stereo.dataloader.kitti import obj_utils


def calc_vector_dist_box_centres(box_centres):

    vector_list = []
    for main_centre in box_centres:
        vector_dist = np.array(box_centres) - np.array(main_centre)
        vector_list.append(vector_dist)

    return vector_list


def get_image_crops(boxes, rgb_img, resize_dims):

    crops = []
    rounded_boxes = np.round(boxes).astype(np.int32)
    for box in rounded_boxes:
        crop = rgb_img[box[0]:box[2], box[1]:box[3]]
        if resize_dims is not None:
            crop = cv2.resize(crop, tuple(resize_dims), interpolation=cv2.INTER_NEAREST)
        crops.append(crop)

    return crops


def greedy_matching(score_matrix, matches):

    # Determine highest SSIM
    max_scores = np.max(score_matrix, axis=1)

    # Match the highest SSIM
    next_box = np.argmax(max_scores)
    matched_box = np.argmax(score_matrix[next_box, :])

    matches[next_box] = matched_box

    # Set used boxes as zero
    score_matrix[next_box, :] = 0
    score_matrix[:, matched_box] = 0

    return score_matrix, matches


def find_matches(score_matrix, mask):

    # Allocate variables
    min_len = min(score_matrix.shape[0], score_matrix.shape[1])
    matches = np.zeros([min_len], dtype=np.int32)

    # Apply mask
    score_matrix = score_matrix * mask

    # Use greedy algorithm
    for i in range(min_len):
        score_matrix, matches = greedy_matching(score_matrix, matches)

    return matches


def main():
    """Matches bboxes from left and right images, and optionally saves them.
    """

    ##############################
    # Options
    ##############################

    score_thresh = [0.5]

    dataset = DatasetBuilder.build_kitti_dataset(
        # DatasetBuilder.KITTI_TRAIN
        DatasetBuilder.KITTI_VAL
        # DatasetBuilder.KITTI_VAL_HALF
        # DatasetBuilder.KITTI_TRAINVAL
        # DatasetBuilder.KITTI_TEST
    )

    obj_type = 'car'
    # obj_type = 'ped'
    # obj_type = 'cyc'

    resize_dims = [128, 128]

    use_masks = True
    dist_thresh = 120.0
    height_thresh = 25.0

    det_method = 'mscnn'

    save_matching_labels = True
    ##############################
    # End of Options
    ##############################

    data_split = dataset.data_split

    # Directories
    detections_dir = oc_stereo.data_dir() + '/detections/{}/kitti_fmt/{}/'.format(det_method,
                                                                                  data_split)
    label_2_dir = detections_dir + '{}_2_{}/data'.format(obj_type, '_'.join(map(str,
                                                                                score_thresh)))
    label_3_dir = detections_dir + '{}_3_{}/data'.format(obj_type, '_'.join(map(str,
                                                                                score_thresh)))

    if obj_type == 'car':
        det_type = 'Car'
    elif obj_type == 'ped':
        det_type = 'Pedestrian'
    elif obj_type == 'cyc':
        det_type = 'Cyclist'
    else:
        raise ValueError('Invalid class')

    # Make directories
    output_label_dir_2 = oc_stereo.data_dir() \
        + '/detections/{}/kitti_fmt/{}_matching_{}/{}_2_matching_{}/data'.format(
        det_method, data_split, '_'.join(map(str, score_thresh)), obj_type, '_'.join(
            map(str, score_thresh)))
    output_label_dir_3 = oc_stereo.data_dir() \
        + '/detections/{}/kitti_fmt/{}_matching_{}/{}_3_matching_{}/data'.format(
        det_method, data_split, '_'.join(map(str, score_thresh)), obj_type, '_'.join(
            map(str, score_thresh)))

    if os.path.isdir(output_label_dir_2):
        pass
    else:
        os.makedirs(output_label_dir_2)

    if os.path.isdir(output_label_dir_3):
        pass
    else:
        os.makedirs(output_label_dir_3)

    num_samples = dataset.num_samples
    sample_names = []
    total_time = []
    for sample_idx in range(num_samples):

        sys.stdout.write('\r{} / {}'.format(sample_idx + 1, num_samples))

        sample_name = dataset.sample_list[sample_idx].name

        mscnn_labels_2 = obj_utils.read_labels(label_2_dir, sample_name)
        mscnn_labels_3 = obj_utils.read_labels(label_3_dir, sample_name)

        output_path_2 = output_label_dir_2 + '/{}.txt'.format(sample_name)
        output_path_3 = output_label_dir_3 + '/{}.txt'.format(sample_name)

        if len(mscnn_labels_2) == 0 or len(mscnn_labels_3) == 0:
            np.savetxt(output_path_2, [],
                       newline='\r\n', fmt='%s')

            np.savetxt(output_path_3, [],
                       newline='\r\n', fmt='%s')
            continue

        # Get RGB image
        img2_path = dataset.get_image_2_path(sample_name)
        img3_path = dataset.get_image_3_path(sample_name)
        bgr_img2 = cv2.imread(img2_path)
        img2 = bgr_img2[..., :: -1]
        bgr_img3 = cv2.imread(img3_path)
        img3 = bgr_img3[..., :: -1]

        # Get 2D bounding boxes [y1, x1, y2, x2]
        boxes_img2 = obj_utils.boxes_2d_from_obj_labels(mscnn_labels_2)
        boxes_img3 = obj_utils.boxes_2d_from_obj_labels(mscnn_labels_3)

        init_time = time.time()

        if use_masks:

            # Calculate height of bounding boxes
            img2_boxes_height = (boxes_img2[:, 2] - boxes_img2[:, 0])
            img3_boxes_height = (boxes_img3[:, 2] - boxes_img3[:, 0])

            # Calculate centres of bounding boxes
            img2_boxes_h_centres = (boxes_img2[:, 3] + boxes_img2[:, 1]) / 2.
            img3_boxes_h_centres = (boxes_img3[:, 3] + boxes_img3[:, 1]) / 2.

            # Create a matrix of height and centre comparison
            height_matrix = np.abs(img2_boxes_height[:, np.newaxis] - img3_boxes_height)
            centre_matrix = np.abs(img2_boxes_h_centres[:, np.newaxis] - img3_boxes_h_centres)

            # Create a mask to help with histogram scoring
            height_mask = height_matrix < height_thresh
            centre_mask = centre_matrix < dist_thresh

            mask = height_mask & centre_mask

            # Remove boxes with no matches
            boxes_img2 = boxes_img2[~np.all(mask == 0, axis=1)]
            boxes_img3 = boxes_img3[~np.all(mask == 0, axis=0)]

            # Re-create height mask
            mask = mask[~np.all(mask == 0, axis=1)]
            mask = mask.T[~np.all(mask == 0, axis=0)]
            mask = mask.T

            num_boxes_img2 = len(boxes_img2)
            num_boxes_img3 = len(boxes_img3)

        else:
            num_boxes_img2 = len(boxes_img2)
            num_boxes_img3 = len(boxes_img3)
            mask = np.ones([num_boxes_img2, num_boxes_img3])

        # Img 2 bounding box crops
        img2_crops = get_image_crops(boxes_img2, img2, resize_dims=resize_dims)
        img3_crops = get_image_crops(boxes_img3, img3, resize_dims=resize_dims)

        # Compute SSIM
        scores = np.zeros([num_boxes_img2, num_boxes_img3])
        for img2_idx, img2_crop in enumerate(img2_crops):
            for img3_idx, img3_crop in enumerate(img3_crops):
                if scores[img2_idx, img3_idx] != 0:
                    continue
                score = measure.compare_ssim(img2_crop, img3_crop, multichannel=True)

                scores[img2_idx, img3_idx] = score

        # Compare histograms
        if num_boxes_img2 < num_boxes_img3:

            matches = find_matches(scores, mask)
            matched_boxes_img3 = boxes_img3[matches]
            corr_boxes = zip(boxes_img2, matched_boxes_img3)

        else:
            matches = find_matches(scores.T, mask.T)
            matched_boxes_img2 = boxes_img2[matches]
            corr_boxes = zip(matched_boxes_img2, boxes_img3)

        time_elapsed = time.time() - init_time
        total_time.append(time_elapsed)

        if save_matching_labels:

            num_dets = len(list(copy.deepcopy(corr_boxes)))

            # Fill other values with empty default values
            det_types = np.reshape([det_type] * num_dets, [num_dets, 1])
            det_trunc_occ = np.full([num_dets, 2], -1, dtype=np.int32)
            det_alphas = np.full([num_dets, 1], -10, dtype=np.int32)
            det_boxes_2d = np.full([num_dets, 4], -1000, dtype=np.int32)
            det_lwh = np.full([num_dets, 3], -1, dtype=np.int32)
            det_xyz = np.full([num_dets, 3], -1000, dtype=np.int32)
            det_ry = np.full([num_dets, 1], -10, dtype=np.int32)
            det_scores = np.full([num_dets, 1], -10, dtype=np.int32)

            matching_box_l = np.column_stack([det_types, det_trunc_occ, det_alphas,
                                              det_boxes_2d,
                                              det_lwh, det_xyz, det_ry,
                                              det_scores])

            matching_box_r = copy.deepcopy(matching_box_l)

            for corr_idx, (box_l, box_r) in enumerate(corr_boxes):

                # Find matching label for box left
                for idx, mscnn_label in enumerate(mscnn_labels_2):

                    if (box_l[0] == mscnn_label.y1 and box_l[1] == mscnn_label.x1
                            and box_l[2] == mscnn_label.y2 and box_l[3] == mscnn_label.x2):

                        matching_box_l[corr_idx, 4:8] = [mscnn_label.x1, mscnn_label.y1,
                                                         mscnn_label.x2, mscnn_label.y2]
                        matching_box_l[corr_idx, 15] = mscnn_label.score

                    else:
                        pass

                # Find matching label for box right
                for idx, mscnn_label in enumerate(mscnn_labels_3):

                    if (box_r[0] == mscnn_label.y1 and box_r[1] == mscnn_label.x1
                            and box_r[2] == mscnn_label.y2 and box_r[3] == mscnn_label.x2):

                        matching_box_r[corr_idx, 4:8] = [mscnn_label.x1, mscnn_label.y1,
                                                         mscnn_label.x2, mscnn_label.y2]
                        matching_box_r[corr_idx, 15] = mscnn_label.score
                    else:
                        pass

            # Skip empty
            if len(matching_box_l) == 0:
                np.savetxt(output_path_2, [],
                           newline='\r\n', fmt='%s')

                np.savetxt(output_path_3, [],
                           newline='\r\n', fmt='%s')
                continue

            # Save to folder
            np.savetxt(output_path_2, matching_box_l,
                       newline='\r\n', fmt='%s')

            np.savetxt(output_path_3, matching_box_r,
                       newline='\r\n', fmt='%s')

            sample_names.append(sample_name)

    if save_matching_labels:
        # Create split file
        with open(os.path.expanduser(
                '~/Kitti/object/{}_matching_{}.txt'.format(
                    data_split, '_'.join(map(str, score_thresh)))), 'w') as f:
            for sample_name in sample_names:
                f.write("%s\n" % sample_name)

    print('Average time', np.mean(total_time))


if __name__ == '__main__':
    main()
