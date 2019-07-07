import copy

import cv2
import numpy as np

from oc_stereo.core import transform_utils
from oc_stereo.dataloader.kitti import calib_utils, depth_map_utils


def read_instance_image(instance_image_path):
    instance_image = cv2.imread(instance_image_path, cv2.IMREAD_GRAYSCALE)
    return instance_image


def get_sr_crop(sample_name, sample_idx, image_dir):
    image_path = image_dir + '/{}.png'.format(sample_name + '_' + str(sample_idx))
    image = cv2.imread(image_path)
    return image


def pad_matching_boxes(left_box, right_box, minimum_shape, image_shape):

    # Parse shape information
    image_height = image_shape[0]
    image_width = image_shape[1]

    min_height = minimum_shape[0]
    min_width = minimum_shape[1]

    left_box_height = left_box[2] - left_box[0]
    left_box_width = left_box[3] - left_box[1]

    right_box_height = right_box[2] - right_box[0]
    right_box_width = right_box[3] - right_box[1]

    # Parse box coordinates
    left_y1 = left_box[0]
    left_x1 = left_box[1]
    left_y2 = left_box[2]
    left_x2 = left_box[3]

    right_y1 = right_box[0]
    right_x1 = right_box[1]
    right_y2 = right_box[2]
    right_x2 = right_box[3]

    # Calculate new height and width using scale factor
    left_height_diff = min_height - left_box_height
    left_width_diff = min_width - left_box_width

    right_height_diff = min_height - right_box_height
    right_width_diff = min_width - right_box_width

    # Pad by the biggest difference
    height_diff = np.maximum(left_height_diff, right_height_diff)
    width_diff = np.maximum(left_width_diff, right_width_diff)

    # Make the the difference rounded to even
    def round_even(n):
        answer = round(n)
        if not answer % 2:
            return answer
        else:
            return answer + 1

    height_diff = round_even(height_diff)
    width_diff = round_even(width_diff)

    if height_diff > 0:
        new_left_y1 = np.clip(left_y1 - height_diff / 2.0, 0.0, None)
        new_right_y1 = np.clip(right_y1 - height_diff / 2.0, 0.0, None)

        new_left_y2 = np.clip(left_y2 + height_diff / 2.0, None, image_height)
        new_right_y2 = np.clip(right_y2 + height_diff / 2.0, None, image_height)

    else:
        new_left_y1 = left_y1
        new_left_y2 = left_y2

        new_right_y1 = right_y1
        new_right_y2 = right_y2

    if width_diff > 0:
        new_left_x1 = np.clip(left_x1 - width_diff / 2.0, 0.0, None)
        new_right_x1 = np.clip(right_x1 - width_diff / 2.0, 0.0, None)

        new_left_x2 = np.clip(left_x2 + width_diff / 2.0, None, image_width)
        new_right_x2 = np.clip(right_x2 + width_diff / 2.0, None, image_width)

    else:
        new_left_x1 = left_x1
        new_left_x2 = left_x2

        new_right_x1 = right_x1
        new_right_x2 = right_x2

    new_box_left = np.array([new_left_y1, new_left_x1, new_left_y2, new_left_x2], dtype=np.int32)
    new_box_right = np.array([new_right_y1, new_right_x1, new_right_y2, new_right_x2],
                             dtype=np.int32)

    return new_box_left, new_box_right


def unpad_box(left_crop, right_crop, left_box, right_box, minimum_shape, image_shape, scale_factor):

    # Parse shape information
    image_height = image_shape[0]
    image_width = image_shape[1]

    min_height = minimum_shape[0]
    min_width = minimum_shape[1]

    left_box_height = left_box[2] - left_box[0]
    left_box_width = left_box[3] - left_box[1]

    right_box_height = right_box[2] - right_box[0]
    right_box_width = right_box[3] - right_box[1]

    # Parse box coordinates
    left_y1 = left_box[0]
    left_x1 = left_box[1]
    left_y2 = left_box[2]
    left_x2 = left_box[3]

    right_y1 = right_box[0]
    right_x1 = right_box[1]
    right_y2 = right_box[2]
    right_x2 = right_box[3]

    # Calculate new height and width
    left_height_diff = min_height - left_box_height
    left_width_diff = min_width - left_box_width

    right_height_diff = min_height - right_box_height
    right_width_diff = min_width - right_box_width

    # Pad by the biggest difference
    height_diff = np.maximum(left_height_diff, right_height_diff)
    width_diff = np.maximum(left_width_diff, right_width_diff)

    # Make the the difference rounded to even
    def round_even(n):
        answer = round(n)
        if not answer % 2:
            return answer
        else:
            return answer + 1

    height_diff = round_even(height_diff)
    width_diff = round_even(width_diff)

    if height_diff > 0:
        new_left_y1 = np.clip(left_y1 - height_diff / 2.0, 0.0, None)
        new_right_y1 = np.clip(right_y1 - height_diff / 2.0, 0.0, None)

        new_left_y2 = np.clip(left_y2 + height_diff / 2.0, None, image_height)
        new_right_y2 = np.clip(right_y2 + height_diff / 2.0, None, image_height)

        # Calculate trims
        left_bottom_trim = -1 * np.int32(np.abs(new_left_y1 - left_y1)) * scale_factor
        left_top_trim = np.int32(np.abs(new_left_y2 - left_y2)) * scale_factor

        right_bottom_trim = -1 * np.int32(np.abs(new_right_y1 - right_y1)) * scale_factor
        right_top_trim = np.int32(np.abs(new_right_y2 - right_y2)) * scale_factor

        if left_bottom_trim == 0:
            left_bottom_trim = None
        if left_top_trim == 0:
            left_top_trim = None

        if right_bottom_trim == 0:
            right_bottom_trim = None
        if right_top_trim == 0:
            right_top_trim = None

    else:
        left_bottom_trim = None
        left_top_trim = None
        right_bottom_trim = None
        right_top_trim = None

    if width_diff > 0:
        new_left_x1 = np.clip(left_x1 - width_diff / 2.0, 0.0, None)
        new_right_x1 = np.clip(right_x1 - width_diff / 2.0, 0.0, None)

        new_left_x2 = np.clip(left_x2 + width_diff / 2.0, None, image_width)
        new_right_x2 = np.clip(right_x2 + width_diff / 2.0, None, image_width)

        # Calculate trims
        left_left_trim = np.int32(np.abs(new_left_x1 - left_x1)) * scale_factor
        left_right_trim = -1 * np.int32(np.abs(new_left_x2 - left_x2)) * scale_factor

        right_left_trim = np.int32(np.abs(new_right_x1 - right_x1)) * scale_factor
        right_right_trim = -1 * np.int32(np.abs(new_right_x2 - right_x2)) * scale_factor

        if left_left_trim == 0:
            left_left_trim = None
        if left_right_trim == 0:
            left_right_trim = None

        if right_left_trim == 0:
            right_left_trim = None
        if right_right_trim == 0:
            right_right_trim = None

    else:
        left_left_trim = None
        left_right_trim = None

        right_left_trim = None
        right_right_trim = None

    unpadded_box_left = left_crop[left_top_trim: left_bottom_trim, left_left_trim: left_right_trim]
    unpadded_box_right = right_crop[right_top_trim: right_bottom_trim, right_left_trim:
                                                                       right_right_trim]

    return unpadded_box_left, unpadded_box_right


# def unpad_box(left_crop, right_crop, box_left, box_right, minimum_shape, image_shape, scale_factor):
#
#     # Parse shape information
#     image_height = image_shape[0]
#     image_width = image_shape[1]
#
#     min_height = minimum_shape[0]
#     min_width = minimum_shape[1]
#
#     box_height = box[2] - box[0]
#     box_width = box[3] - box[1]
#
#     # Parse box coordinates
#     y1 = box[0]
#     x1 = box[1]
#     y2 = box[2]
#     x2 = box[3]
#
#     # Calculate new height and width using scale factor
#     height_diff = min_height - box_height
#     width_diff = min_width - box_width
#
#     if height_diff >= 0:
#         new_y1 = np.clip(y1 - height_diff / 2.0, 0.0, None)
#         new_y2 = np.clip(y2 + height_diff / 2.0, None, image_height)
#
#         bottom_trim = -1 * np.int32(np.abs(new_y1 - y1)) * scale_factor
#         top_trim = np.int32(np.abs(new_y2 - y2)) * scale_factor
#
#         if bottom_trim == 0:
#             bottom_trim = None
#         if top_trim == 0:
#             top_trim = None
#
#     else:
#         bottom_trim = None
#         top_trim = None
#
#     if width_diff >= 0:
#         new_x1 = np.clip(x1 - width_diff / 2.0, 0.0, None)
#         new_x2 = np.clip(x2 + width_diff / 2.0, None, image_width)
#
#         left_trim = np.int32(np.abs(new_x1 - x1)) * scale_factor
#         right_trim = -1 * np.int32(np.abs(new_x2 - x2)) * scale_factor
#
#         if left_trim == 0:
#             left_trim = None
#         if right_trim == 0:
#             right_trim = None
#
#     else:
#         left_trim = None
#         right_trim = None
#
#     unpadded_img_crop = img_crop[top_trim: bottom_trim, left_trim: right_trim]
#
#     # unpadded_img_crop = img_crop[top_trim:(img_crop_height - bottom_trim),
#     #                              left_trim: (img_crop_width - right_trim)]
#
#     return unpadded_img_crop


def get_instance_image(sample_name, instance_dir):

    instance_image_path = instance_dir + '/{}.png'.format(sample_name)
    instance_image = read_instance_image(instance_image_path)

    return instance_image


def get_instance_mask_list(instance_img, num_instances=None):
    """Creates n-dimensional image from instance image with one channel per instance

    Args:
        instance_img: (H, W) instance image
        num_instances: (optional) number of instances in the image. If None, will use the
            highest value pixel as the number of instances, but may miss the last
            instances if they have no points.

    Returns:
        instance_masks: (k, H, W) instance masks where k is the unique values of the instance im
    """

    if num_instances is None:
        valid_pixels = instance_img[instance_img != 255]
        if len(valid_pixels) == 0:
            return []
        num_instances = np.max(valid_pixels) + 1

    instance_masks = np.asarray([(instance_img == instance_idx)
                                 for instance_idx in range(num_instances)])
    return instance_masks


def read_instance_maps(instance_maps_path):
    return np.load(instance_maps_path)


def get_valid_inst_box_2d_crop(box_2d, input_map):
    """Gets a valid 2D box crop. If the box is too small, it returns a single pixel.

    Args:
        box_2d: 2D box
        input_map: (H, W, C) Input map

    Returns:
        inst_box_2d_crop: Crop of input map
    """

    # Get box dimensions
    box_2d_rounded = np.round(box_2d).astype(np.int32)
    box_2d_rounded_h = box_2d_rounded[2] - box_2d_rounded[0]
    box_2d_rounded_w = box_2d_rounded[3] - box_2d_rounded[1]

    # Check dimensions
    if box_2d_rounded_h > 0 and box_2d_rounded_w > 0:
        # Crop if valid box
        inst_box_2d_crop = input_map[box_2d_rounded[0]:box_2d_rounded[2],
                                     box_2d_rounded[1]:box_2d_rounded[3]]
    else:
        # Invalid box, use single pixel
        inst_box_2d_crop = input_map[
            box_2d_rounded[0]:box_2d_rounded[0] + 1,
            box_2d_rounded[1]:box_2d_rounded[1] + 1]

    return inst_box_2d_crop


def np_instance_crop(boxes_2d, boxes_3d, instance_masks, input_map, roi_size,
                     view_norm=False, cam_p=None, viewing_angles=None,
                     centroid_type='bottom', rotate_view=True):
    """Crops an input map for an instance

    Args:
        boxes_2d: (N, 4) 2D boxes [y1, x1, y2, x2]
        boxes_3d: (N, 6) 3D boxes
        instance_masks:  (N, H, W) boolean instance masks
        input_map: (H, W, C) Input map with C channels. Should be in camN frame.
        roi_size: roi crop size [h, w]
        view_norm: (optional) Apply view normalization for xyz maps
        cam_p: (3, 4) Camera projection matrix
        viewing_angles: (N) Viewing angles
        centroid_type (string): centroid position (bottom or middle)
        rotate_view: bool whether to rotating by viewing angle

    Returns:
        all_instance_xyz: (N, roi_h, roi_w, C) cropped and resized instance map
        valid_pixel_mask: (N, roi_h, roi_w, 1) mask of valid pixels
    """
    # TODO: Add unit tests, fix valid pixel mask return

    input_map_shape = input_map.shape

    if len(input_map_shape) != 3:
        raise ValueError('Invalid input_map_shape', input_map_shape)

    all_instance_maps = []
    all_valid_mask_maps = []
    for instance_idx, (instance_mask, box_2d, box_3d) in enumerate(
            zip(instance_masks, boxes_2d, boxes_3d)):

        # Apply instance mask
        input_map_masked = instance_mask[:, :, np.newaxis] * input_map

        # Crop and resize
        inst_box_2d_crop = get_valid_inst_box_2d_crop(box_2d, input_map_masked)
        instance_map_resized = cv2.resize(inst_box_2d_crop, tuple(roi_size),
                                          interpolation=cv2.INTER_NEAREST)

        # Calculate valid mask, works for both point clouds and RGB
        instance_map_resized_shape = instance_map_resized.shape
        if len(instance_map_resized_shape) == 3:
            valid_mask_map = np.sum(abs(instance_map_resized), axis=2) > 0.1
        else:
            valid_mask_map = abs(instance_map_resized) > 0.1
        all_valid_mask_maps.append(valid_mask_map)

        if view_norm:
            if input_map.shape[2] != 3:
                raise ValueError('Invalid shape to apply view normalization')

            # Get viewing angle and rotation matrix
            viewing_angle = viewing_angles[instance_idx]

            # Get x offset (b_cam) from calibration: cam_mat[0, 3] = (-f_x * b_cam)
            x_offset = -cam_p[0, 3] / cam_p[0, 0]
            cam0_centroid = box_3d[0:3]
            camN_centroid = cam0_centroid - [x_offset, 0, 0]

            if centroid_type == 'middle':
                # Move centroid to half the box height
                half_h = box_3d[5] / 2.0
                camN_centroid[1] -= half_h

            if rotate_view:
                inst_xyz_map_local = apply_view_norm_to_pc_map(
                    instance_map_resized, valid_mask_map, viewing_angle, camN_centroid,
                    roi_size)
            else:
                inst_xyz_map_local = apply_view_norm_to_pc_map(
                    instance_map_resized, valid_mask_map, 0.0, camN_centroid, roi_size)

            all_instance_maps.append(inst_xyz_map_local)

        else:
            all_instance_maps.append(instance_map_resized)

    return np.asarray(all_instance_maps), np.asarray(all_valid_mask_maps)


def np_instance_xyz_crop(boxes_2d, boxes_3d, instance_masks, xyz_map, roi_size,
                         view_norm=False, cam_p=None, viewing_angles=None,
                         centroid_type='bottom', rotate_view=True):
    """Crops an input map for an instance

    Args:
        boxes_2d: (N, 4) 2D boxes [y1, x1, y2, x2]
        boxes_3d: (N, 6) 3D boxes
        instance_masks:  (N, H, W) boolean instance masks
        xyz_map: (H, W, C) Input map with C channels. Should be in camN frame.
        roi_size: roi crop size [h, w]
        view_norm: (optional) Apply view normalization for xyz maps
        cam_p: (3, 4) Camera projection matrix
        viewing_angles: (N) Viewing angles
        centroid_type (string): centroid position (bottom or middle)
        rotate_view: bool whether to rotating by viewing angle

    Returns:
        all_instance_xyz: (N, roi_h, roi_w, C) cropped and resized instance map
        valid_pixel_mask: (N, roi_h, roi_w, 1) mask of valid pixels
    """
    # TODO: Add unit tests, fix valid pixel mask return

    input_map_shape = xyz_map.shape

    if len(input_map_shape) != 3:
        raise ValueError('Invalid input_map_shape', input_map_shape)

    all_instance_maps = []
    all_valid_mask_maps = []
    for instance_idx, (instance_mask, box_2d, box_3d) in enumerate(
            zip(instance_masks, boxes_2d, boxes_3d)):

        # Apply instance mask
        input_map_masked = instance_mask[:, :, np.newaxis] * xyz_map

        # Crop and resize
        inst_box_2d_crop = get_valid_inst_box_2d_crop(box_2d, input_map_masked)
        instance_map_resized = cv2.resize(inst_box_2d_crop, tuple(roi_size),
                                          interpolation=cv2.INTER_NEAREST)

        # Calculate valid mask, works for both point clouds and RGB
        instance_map_resized_shape = instance_map_resized.shape
        if len(instance_map_resized_shape) == 3:
            valid_mask_map = np.sum(abs(instance_map_resized), axis=2) > 0.1
        else:
            valid_mask_map = abs(instance_map_resized) > 0.1
        all_valid_mask_maps.append(valid_mask_map)

        if view_norm:
            if xyz_map.shape[2] != 3:
                raise ValueError('Invalid shape to apply view normalization')

            # Get viewing angle and rotation matrix
            viewing_angle = viewing_angles[instance_idx]

            # Get x offset (b_cam) from calibration: cam_mat[0, 3] = (-f_x * b_cam)
            x_offset = -cam_p[0, 3] / cam_p[0, 0]
            cam0_centroid = box_3d[0:3]
            camN_centroid = cam0_centroid - [x_offset, 0, 0]

            if centroid_type == 'middle':
                # Move centroid to half the box height
                half_h = box_3d[5] / 2.0
                camN_centroid[1] -= half_h

            inst_pc_map = instance_map_resized.transpose([2, 0, 1])

            if rotate_view:
                inst_xyz_map_local = apply_view_norm_to_pc_map(
                    inst_pc_map, valid_mask_map, viewing_angle, camN_centroid,
                    roi_size)
            else:
                inst_xyz_map_local = apply_view_norm_to_pc_map(
                    inst_pc_map, valid_mask_map, 0.0, camN_centroid, roi_size)

            all_instance_maps.append(inst_xyz_map_local)

        else:
            all_instance_maps.append(instance_map_resized)

    return np.asarray(all_instance_maps), np.asarray(all_valid_mask_maps)


def np_instance_xyz_crop_from_depth_map(boxes_2d, boxes_3d, instance_masks,
                                        depth_map, roi_size, cam_p, viewing_angles,
                                        use_pixel_centres, use_corr_factors, centroid_type='bottom',
                                        rotate_view=True):
    """Crops the depth map for an instance and returns local instance xyz crops

    Args:
        boxes_2d: (N, 4) List of 2D boxes [y1, x1, y2, x2]
        boxes_3d: (N, 6) 3D boxes
        instance_masks: (N, H, W) Boolean instance masks
        depth_map: (H, W) Depth map
        roi_size: ROI crop size [h, w]
        cam_p: (3, 4) Camera projection matrix
        viewing_angles: (N) Viewing angles
        use_pixel_centres: (optional) If True, re-projects depths such that they will
            project back to the centre of the ROI pixel. Otherwise, they will project
            to the top left corner.
        use_corr_factors: (optional) If True, applies correction factors along xx and yy
            according to depth in order to reduce projection error.
        centroid_type (string): centroid position (bottom or middle)
        rotate_view: bool whether to rotate by viewing angle

    Returns:
        xyz_out: (N, roi_h, roi_w, 3) instance xyz map in local coordinate frame
        valid_pixel_mask: (N, roi_h, roi_w, 1) mask of valid pixels
    """

    depth_map_shape = depth_map.shape
    if len(depth_map_shape) != 2:
        raise ValueError('Invalid depth_map_shape', depth_map_shape)

    all_inst_depth_crops, all_inst_valid_masks = np_instance_crop(
        boxes_2d=boxes_2d,
        boxes_3d=boxes_3d,
        instance_masks=instance_masks,
        input_map=np.expand_dims(depth_map, 2),
        roi_size=roi_size,
        view_norm=False)

    camN_inst_pc_maps = [depth_map_utils.depth_patch_to_pc_map(
        inst_depth_crop, box_2d, cam_p, roi_size, depth_map_shape=depth_map.shape[0:2],
        use_pixel_centres=use_pixel_centres, use_corr_factors=use_corr_factors)
        for inst_depth_crop, box_2d in zip(all_inst_depth_crops, boxes_2d)]

    # Get x offset (b_cam) from calibration: cam_mat[0, 3] = (-f_x * b_cam)
    x_offset = -cam_p[0, 3] / cam_p[0, 0]
    camN_centroids = boxes_3d[:, 0:3] - [x_offset, 0, 0]

    if centroid_type == 'middle':
        # Move centroid to half the box height
        half_h = boxes_3d[:, 5] / 2.0
        camN_centroids[:, 1] -= half_h

    if not rotate_view:
        viewing_angles = np.zeros_like(viewing_angles)

    inst_xyz_maps_local = [
        apply_view_norm_to_pc_map(inst_pc_map, valid_mask, viewing_angle, centroid, roi_size)
        for inst_pc_map, valid_mask, viewing_angle, centroid in zip(
            camN_inst_pc_maps, all_inst_valid_masks, viewing_angles, camN_centroids)]

    return inst_xyz_maps_local, all_inst_valid_masks


def apply_view_norm_to_pc_map(inst_pc_map, valid_mask_map, viewing_angle, centroid, roi_size):
    """Applies view normalization on instance pc map

    Args:
        inst_pc_map: (3, H, W) Instance pc map
        valid_mask_map: (H, W) Valid pixel mask
        viewing_angle: Viewing angle
        centroid: Centroid [x, y, z]
        roi_size: ROI size [h, w]

    Returns:
        inst_xyz_map: (H, W, 3) View normalized xyz map
    """

    # Apply view normalization
    tr_mat = transform_utils.np_get_tr_mat(-viewing_angle, -centroid)

    # Move to origin
    inst_pc_padded = transform_utils.pad_pc(inst_pc_map.reshape(3, -1))
    inst_pc_local = tr_mat.dot(inst_pc_padded)[0:3]

    inst_xyz_map = np.reshape(inst_pc_local.T, (roi_size[0], roi_size[1], 3))
    inst_xyz_map = inst_xyz_map * np.expand_dims(valid_mask_map, 2)

    return inst_xyz_map


def inst_points_global_to_local(inst_points_global, viewing_angle, centroid):
    """Converts global points to local points in same camera frame."""

    # Apply view normalization
    rot_mat = transform_utils.np_get_tr_mat(-viewing_angle, -centroid)

    # Rotate, then translate
    inst_pc_padded = transform_utils.pad_pc(inst_points_global.T)
    inst_pc_local = rot_mat.dot(inst_pc_padded)[0:3]

    return inst_pc_local.T


def inst_points_local_to_global(inst_points_local, viewing_angle, centroid):
    """Converts local points to global points in same camera frame"""

    # Rotate predicted instance points to viewing angle and translate to guessed centroid
    rot_mat = transform_utils.np_get_tr_mat(viewing_angle, (0.0, 0.0, 0.0))
    t_mat = transform_utils.np_get_tr_mat(0.0, centroid)

    inst_points_rotated = transform_utils.apply_tr_mat_to_points(
        rot_mat, inst_points_local)

    inst_points_global = transform_utils.apply_tr_mat_to_points(t_mat, inst_points_rotated)

    return inst_points_global


def get_exp_proj_uv_map(box_2d, roi_size, round_box_2d=False, use_pixel_centres=False):
    """Get expected grid projection of a 2D box based on roi size, if pixels are evenly spaced.
    Points project to the top left of each pixel.

    Args:
        box_2d: 2D box
        roi_size: ROI size [h, w]
        use_pixel_centres: (optional) If True, return projections to centre of pixels

    Returns:
        proj_uv_map: (H, W, 2) Expected box_2d projection uv map
    """

    # Grid start and stop
    if round_box_2d:
        inst_u1, inst_u2 = np.round(box_2d[[1, 3]])
        inst_v1, inst_v2 = np.round(box_2d[[0, 2]])
    else:
        inst_u1, inst_u2 = box_2d[[1, 3]]
        inst_v1, inst_v2 = box_2d[[0, 2]]

    # Grid spacing
    roi_h, roi_w = roi_size
    grid_u_spacing = (inst_u2 - inst_u1) / float(roi_w)
    grid_v_spacing = (inst_v2 - inst_v1) / float(roi_h)

    if use_pixel_centres:

        # Grid along u
        grid_u_half_spacing = grid_u_spacing / 2.0
        grid_u = np.linspace(
            inst_u1 + grid_u_half_spacing,
            inst_u2 - grid_u_half_spacing,
            roi_w)

        # Grid along v
        grid_v_half_spacing = grid_v_spacing / 2.0
        grid_v = np.linspace(
            inst_v1 + grid_v_half_spacing,
            inst_v2 - grid_v_half_spacing,
            roi_h)

        proj_uv_map = np.meshgrid(grid_u, grid_v)

    else:
        # Use linspace instead of arange to avoid including last value
        grid_u = np.linspace(inst_u1, inst_u2 - grid_u_spacing, roi_w, dtype=np.float32)
        grid_v = np.linspace(inst_v1, inst_v2 - grid_v_spacing, roi_h, dtype=np.float32)

        proj_uv_map = np.meshgrid(grid_u, grid_v)

    return np.dstack(proj_uv_map)


def proj_points(xz_dist, centroid_y, viewing_angle, cam2_inst_points_local,
                cam_p, rotate_view=True):
    """Projects point based on estimated transformation matrix
    calculated from xz_dist and viewing angle

    Args:
        xz_dist: distance along viewing angle
        centroid_y: box centroid y
        viewing_angle: viewing angle
        cam2_inst_points_local: (N, 3) instance points
        cam_p: (3, 4) camera projection matrix
        rotate_view: bool whether to rotate by viewing angle

    Returns:
        points_uv: (2, N) The points projected in u, v coordinates
        valid_points_mask: (N) Mask of valid points
    """

    guess_x = xz_dist * np.sin(viewing_angle)
    guess_y = centroid_y
    guess_z = xz_dist * np.cos(viewing_angle)

    # Rotate predicted instance points to viewing angle and translate to guessed centroid
    rot_mat = transform_utils.np_get_tr_mat(viewing_angle, (0.0, 0.0, 0.0))
    t_mat = transform_utils.np_get_tr_mat(0.0, [guess_x, guess_y, guess_z])
    if rotate_view:
        cam2_points_rotated = transform_utils.apply_tr_mat_to_points(
            rot_mat, cam2_inst_points_local)
    else:
        cam2_points_rotated = cam2_inst_points_local

    cam2_points_global = transform_utils.apply_tr_mat_to_points(t_mat, cam2_points_rotated)

    # Get valid points mask
    valid_points_mask = np.sum(np.abs(cam2_points_rotated), axis=1) > 0.1

    # Shift points into cam0 frame for projection
    # Get x offset (b_cam) from calibration: cam_mat[0, 3] = (-f_x * b_cam)
    x_offset = -cam_p[0, 3] / cam_p[0, 0]

    # Shift points from cam2 to cam0 frame
    cam0_points_global = (cam2_points_global + [x_offset, 0, 0]) * valid_points_mask.reshape(-1, 1)

    # Project back to image
    pred_points_in_img = calib_utils.project_pc_to_image(
        cam0_points_global.T, cam_p) * valid_points_mask

    return pred_points_in_img, valid_points_mask


def minimize_proj_error_naive(box_2d, box_3d, roi_size, viewing_angle,
                              valid_points_mask, xz_dist_start, inst_points_local, cam_p,
                              rotate_view=True):
    """

    Args:
        box_2d: 2D box
        box_3d: 3D box
        roi_size: ROI size [h, w]
        viewing_angle: Viewing angle towards box centroid
        valid_points_mask: (N) Mask of valid points
        xz_dist_start: Starting guess for xz_dist minimization
        inst_points_local: (N, 3) Instance points in local coordinate frame
        cam_p: Camera projection matrix
        rotate_view: bool whether to rotate by viewing angle

    Returns:
        best_xz_dist: Best xz distance in camN frame
    """
    exp_grid_uv = get_exp_proj_uv_map(box_2d, roi_size)

    # Reshape valid mask
    valid_points_map_mask = valid_points_mask.reshape(1, *roi_size)

    rot_mat = transform_utils.np_get_tr_mat(viewing_angle, [0, 0, 0])

    # Guess different distances
    best_xz_dist = 0.0
    best_proj_err_uv_norm = None

    for dist in np.arange(xz_dist_start - 3.0, xz_dist_start + 3.0, 0.1):

        # Position in cam2
        guess_x = dist * np.sin(viewing_angle)
        guess_y = box_3d[1]
        guess_z = dist * np.cos(viewing_angle)

        guess_t_mat = transform_utils.np_get_tr_mat(0.0, [guess_x, guess_y, guess_z])

        # Rotate instance points by viewing angle and translate to guess position
        if rotate_view:
            points_rotated = transform_utils.apply_tr_mat_to_points(rot_mat, inst_points_local)
        else:
            points_rotated = inst_points_local

        points_shifted = transform_utils.apply_tr_mat_to_points(
            guess_t_mat, points_rotated)

        # Shift points into cam0
        x_offset = -cam_p[0, 3] / cam_p[0, 0]
        cam0_points_shifted = (points_shifted + [x_offset, 0, 0]) * valid_points_mask.reshape(-1, 1)

        # Project back to image
        points_in_img = calib_utils.project_pc_to_image(cam0_points_shifted.T, cam_p)

        # Error for prediction
        proj_grid_uv = points_in_img.reshape(2, *roi_size)
        proj_err_uv = proj_grid_uv - exp_grid_uv

        abs_proj_err_uv = np.abs(proj_err_uv) * valid_points_map_mask

        # # Use inliers
        # min_val, max_val = np.percentile(abs_proj_err_uv, [0, 95])
        # inlier_mask = (abs_proj_err_uv >= min_val) & (abs_proj_err_uv <= max_val)
        # inlier_values = abs_proj_err_uv[inlier_mask]

        # proj_err_norm = proj_err_sum / len(inlier_values)

        proj_err_sum = np.sum(abs_proj_err_uv)
        proj_err_norm = proj_err_sum / np.count_nonzero(abs_proj_err_uv)

        # Save best values
        if best_proj_err_uv_norm is None or proj_err_norm < best_proj_err_uv_norm:
            best_xz_dist = dist
            best_proj_err_uv_norm = proj_err_norm

    return best_xz_dist, best_proj_err_uv_norm


def powerlaw(x, a, m, c):
    return a * x ** m + c


def est_cen_z_from_box_2d(box_2d, class_str, trend_data='kitti'):
    """Rough estimate of centroid depth from height of 2D box using a powerlaw
    """
    box_2d_height = box_2d[2] - box_2d[0]

    if class_str == 'Car':

        if trend_data == 'kitti':
            initial_depth = powerlaw(box_2d_height, 750.0522869, -0.8696792, -0.3700546)

        elif trend_data == 'mscnn':
            initial_depth = powerlaw(box_2d_height, 740.5579893, -0.8674848, -0.4178865)

        else:
            raise ValueError('Invalid box_source', trend_data)

    elif class_str == 'Pedestrian':

        if trend_data == 'kitti':
            initial_depth = powerlaw(box_2d_height, 1044.5557193, -0.9420833, -0.4836576)

        elif trend_data == 'mscnn':
            initial_depth = powerlaw(box_2d_height, 859.1642611, -0.8830018, -1.4823943)

        else:
            raise ValueError('Invalid box_source', trend_data)

    elif class_str == 'Cyclist':

        if trend_data == 'kitti':
            initial_depth = powerlaw(box_2d_height, 956.0591339, -0.9188194, -0.5118353)

        elif trend_data == 'mscnn':
            initial_depth = powerlaw(box_2d_height, 620.7076927, -0.7871749, -3.0151643)

        else:
            raise ValueError('Invalid box_source', trend_data)

    else:
        raise ValueError('Invalid class_str', class_str)

    return initial_depth


def tf_est_cen_z_from_box_2d(boxes_2d, trend_data='kitti'):
    """Rough estimate of centroid depth from height 2D boxes using a powerlaw
    """
    boxes_2d_h = boxes_2d[:, 2] - boxes_2d[:, 0]

    if trend_data == 'kitti':
        initial_depth = powerlaw(boxes_2d_h, 750.0522869, -0.8696792, -0.3700546)

    elif trend_data == 'mscnn':
        initial_depth = powerlaw(boxes_2d_h, 729.7622204, -0.8601836, -0.6222343)

    else:
        raise ValueError('Invalid box_source', trend_data)

    return initial_depth


def est_initial_xz_dist_from_box_2d(box_2d, powerlaw_coeffs='scipy'):
    """Rough estimate of centroid xz distance from height of 2D box using a powerlaw
    """

    box_2d_height = box_2d[2] - box_2d[0]

    # Get initial xz_dist from power law fit
    if powerlaw_coeffs == 'libre':
        initial_xz_dist = powerlaw(box_2d_height, 745.7241270881, -0.8630725176, 0.0)
    elif powerlaw_coeffs == 'scipy':
        initial_xz_dist = powerlaw(box_2d_height, 998.5765217, -0.9525137, 1.5725772)
    else:
        raise ValueError('Invalid powerlaw_coeffs', powerlaw_coeffs)

    return initial_xz_dist


def get_prop_cen_z_offset(class_str):
    """Get the proposal z centroid offset depending on the class.
    """

    if class_str == 'Car':
        offset = 2.17799973487854
    elif class_str == 'Pedestrian':
        offset = 0.351921409368515
    elif class_str == 'Cyclist':
        offset = 0.8944902420043945
    else:
        raise ValueError('Invalid class_str', class_str)

    return offset


def inflate_box_2d(box_2d, image_shape, scale_factor=1.5):
    """Enlarge 2D box by a set factor in each direction. Ensure not extend past image dimensions.

    Args:
        box_2d: [y1, x1, y2, x2]
        image_shape: (h, w) image shape
        scale_factor: scaling factor

    Returns:
        inflated_box_2d: [y1, x1, y2, x2]
    """

    # Image height and width
    image_h = image_shape[0]
    image_w = image_shape[1]

    # Parse old dimensions
    y1 = box_2d[0]
    x1 = box_2d[1]
    y2 = box_2d[2]
    x2 = box_2d[3]

    old_height = y2 - y1
    old_width = x2 - x1

    # Calculate new height and width using scale factor
    new_height = old_height * scale_factor
    new_width = old_width * scale_factor

    # Calculate new dimensions
    height_diff = new_height - old_height
    width_diff = new_width - old_width

    new_y1 = np.clip(y1 - height_diff / 2.0, 0.0, None)
    new_y2 = np.clip(y2 + height_diff / 2.0, None, image_h)

    new_x1 = np.clip(x1 - width_diff / 2.0, 0.0, None)
    new_x2 = np.clip(x2 + width_diff / 2.0, None, image_w)

    return [new_y1, new_x1, new_y2, new_x2]


def make_boxes_same_height(boxes_left, boxes_right, round=False):
    """Makes the left and right boxes have the same y1 and y2

    Args:
        boxes_left: [y1, x1, y2, x2]
        boxes_right: [y1, x1, y2, x2]

    Returns:
        new_boxes_left: boxes with matching y1 and y2 with the right boxes
        new_boxes_right: boxes with matching y1 and y2 with the left boxes
    """

    # Round boxes
    boxes_left = np.int32(boxes_left)
    boxes_right = np.int32(boxes_right)

    # Create a copy of the boxes
    new_boxes_left = copy.deepcopy(boxes_left)
    new_boxes_right = copy.deepcopy(boxes_right)

    for idx, (box_left, box_right) in enumerate(zip(boxes_left, boxes_right)):

        box_left_y2 = box_left[2]
        box_left_y1 = box_left[0]

        box_right_y2 = box_right[2]
        box_right_y1 = box_right[0]

        # Make both boxes have the higher of the two y2 value
        if box_left_y2 > box_right_y2:
            new_boxes_right[idx][2] = box_left_y2
        else:
            new_boxes_left[idx][2] = box_right_y2

        # Make both boxes have the lower of the two y1 values
        if box_left_y1 < box_right_y1:
            new_boxes_right[idx][0] = box_left_y1
        else:
            new_boxes_left[idx][0] = box_right_y1

    return new_boxes_left, new_boxes_right


def make_boxes_square(boxes_left, boxes_right, image_shape, keep_same_y=False, round=True):
    """Makes the bounding boxes square

    Args:
        boxes_left: [y1, x1, y2, x2]
        boxes_right: [y1, x1, y2, x2]
        image_shape: (H, W)
        keep_same_y: Keep the same y1 and y2 coordinates
        round: Whether to round the coordinates

    Returns:
        new_boxes_left: square boxes left
        new_boxes_right: square boxes right
    """

    new_boxes_left = copy.deepcopy(boxes_left)
    new_boxes_right = copy.deepcopy(boxes_right)

    for idx, (box_left, box_right) in enumerate(zip(boxes_left, boxes_right)):

        # Calculate heights and widths
        box_left_height = box_left[2] - box_left[0]
        box_left_width = box_left[3] - box_left[1]

        box_right_height = box_right[2] - box_right[0]
        box_right_width = box_right[3] - box_right[1]

        # Determine the aspect ratio
        ar = box_left_height / box_left_width

        if ar < 1.0:

            diff_left = box_left_width - box_left_height
            diff_right = box_right_width - box_right_height

            if keep_same_y:
                if box_left[2] != box_right[2]:
                    raise ValueError('box_left_y2 {} not equal to box_left_y1 {}'.format(
                        box_left[2], box_right[2]))

                if diff_left > diff_right:
                    # Increase the width as well to make up for extra height addition
                    new_boxes_right[idx][3] += (diff_left - diff_right) / 2.0
                    new_boxes_right[idx][1] -= (diff_left - diff_right) / 2.0

                    diff_right = diff_left

                else:
                    # diff right > diff_left

                    # Increase the width as well to make up for extra height addition
                    new_boxes_left[idx][3] += (diff_right - diff_left) / 2.0
                    new_boxes_left[idx][1] -= (diff_right - diff_left) / 2.0
                    diff_left = diff_right

            # Increase box heights to match the widths
            new_boxes_left[idx][2] += diff_left / 2.0
            new_boxes_left[idx][0] -= diff_left / 2.0

            # Repeat for box right
            new_boxes_right[idx][2] += diff_right / 2.0
            new_boxes_right[idx][0] -= diff_right / 2.0

        else:
            # Increase box widths to match the height
            diff = box_left_height - box_left_width
            new_boxes_left[idx][3] += diff / 2.0
            new_boxes_left[idx][1] -= diff / 2.0

            # Repeat for box right
            diff = box_right_height - box_right_width
            new_boxes_right[idx][3] += diff / 2.0
            new_boxes_right[idx][1] -= diff / 2.0

    if round:
        new_boxes_left = np.round(new_boxes_left).astype(np.int32)
        new_boxes_right = np.round(new_boxes_right).astype(np.int32)

        # The height dimension of both boxes are the same
        boxes_left_2d_rounded_h = new_boxes_left[:, 2] - new_boxes_left[:, 0]
        boxes_right_2d_rounded_h = new_boxes_right[:, 2] - new_boxes_right[:, 0]

        boxes_left_2d_rounded_w = new_boxes_left[:, 3] - new_boxes_left[:, 1]
        boxes_right_2d_rounded_w = new_boxes_right[:, 3] - new_boxes_right[:, 1]

        # If the boxes aren't square, add the difference to the width to make them square
        diff_left = boxes_left_2d_rounded_h - boxes_left_2d_rounded_w
        new_boxes_left[:, 3] += diff_left

        diff_right = boxes_right_2d_rounded_h - boxes_right_2d_rounded_w
        new_boxes_right[:, 3] += diff_right

    # Check if out of image (left)
    old_boxes = copy.deepcopy(new_boxes_left)
    below_img_2 = new_boxes_left[:, 0] < 0
    if any(below_img_2):
        diff = 0 - new_boxes_left[:, 0][below_img_2]
        new_boxes_left[:, 0][below_img_2] = 0
        new_boxes_left[:, 2][below_img_2] += diff

    left_img_2 = new_boxes_left[:, 1] < 0
    if any(left_img_2):
        diff = 0 - new_boxes_left[:, 1][left_img_2]
        new_boxes_left[:, 1][left_img_2] = 0
        new_boxes_left[:, 3][left_img_2] += diff

    right_img_2 = new_boxes_left[:, 3] > image_shape[1]
    if any(right_img_2):
        diff = image_shape[1] - new_boxes_left[:, 3][right_img_2]
        new_boxes_left[:, 3][right_img_2] = image_shape[1]
        new_boxes_left[:, 1][right_img_2] += diff

    above_img_2 = new_boxes_left[:, 2] > image_shape[0]
    if any(above_img_2):
        diff = image_shape[0] - new_boxes_left[:, 2][above_img_2]
        new_boxes_left[:, 2][above_img_2] = image_shape[0]
        new_boxes_left[:, 0][above_img_2] += diff

    # Repeat for image right
    below_img_3 = new_boxes_right[:, 0] < 0
    if any(below_img_3):
        diff = 0 - new_boxes_right[:, 0][below_img_3]
        new_boxes_right[:, 0][below_img_3] = 0
        new_boxes_right[:, 2][below_img_3] += diff

    left_img_3 = new_boxes_right[:, 1] < 0
    if any(left_img_3):
        diff = 0 - new_boxes_right[:, 1][left_img_3]
        new_boxes_right[:, 1][left_img_3] = 0
        new_boxes_right[:, 3][left_img_3] += diff

    right_img_3 = new_boxes_left[:, 3] > image_shape[1]
    if any(right_img_3):
        diff = image_shape[1] - new_boxes_right[:, 3][right_img_3]
        new_boxes_right[:, 3][right_img_3] = image_shape[1]
        new_boxes_right[:, 1][right_img_3] += diff

    above_img_3 = new_boxes_right[:, 2] > image_shape[0]
    if any(above_img_3):
        diff = image_shape[0] - new_boxes_right[:, 2][above_img_3]
        new_boxes_right[:, 2][above_img_3] = image_shape[0]
        new_boxes_right[:, 0][above_img_3] += diff

    # Check if correct
    check_right = \
        (new_boxes_right[:, 2] - new_boxes_right[:, 0]) == (new_boxes_right[:, 3] -
                                                            new_boxes_right[:, 1])

    check_left = \
        (new_boxes_left[:, 2] - new_boxes_left[:, 0]) == (new_boxes_left[:, 3] -
                                                          new_boxes_left[:, 1])

    if False in check_right or False in check_left:
        raise ValueError('Resulting box is not square')

    return new_boxes_left, new_boxes_right


def calc_i2_prime(roi_size):

    # Calculate local disparity
    grid_space = roi_size[1]
    # TODO: Check if did linspace correct (start with 0)
    i2_prime = np.linspace(0, roi_size[1] - 1, grid_space)
    i2_prime = np.tile(i2_prime, (roi_size[0], 1))

    return i2_prime


def calc_local_disp(coord_grid_u, i2_prime, box_left, box_right, disp_map, roi_size, instance_mask):

    # Crop and resize
    disp_map_crop = get_valid_inst_box_2d_crop(box_left, disp_map)
    disp_map_crop_resized = cv2.resize(disp_map_crop, (roi_size[1], roi_size[0]),
                                       interpolation=cv2.INTER_NEAREST)

    # TODO: Redundant?
    coord_grid_u *= instance_mask

    # Shift coordinate grid by disparity
    coords_img3 = coord_grid_u - disp_map_crop_resized

    # Shift coordinates by the box right edge
    i3 = coords_img3 - box_right[1]

    # Calculate coordinates within roi size crop
    box_right_width = box_right[3] - box_right[1]
    i3_prime = i3 / box_right_width * roi_size[1]

    # Mask
    i3_prime *= instance_mask
    i2_prime *= instance_mask

    local_disp = i2_prime - i3_prime

    return local_disp


def calc_global_from_local_disp(local_disp, coord_u_grid, i2_prime, box_right, roi_size,
                                instance_mask):

    # TODO: Redundant mask?
    i3_prime = (i2_prime - local_disp) * instance_mask

    # Box width
    box_width = box_right[3] - box_right[1]

    i3 = i3_prime / roi_size[1] * box_width

    x3 = i3 + box_right[1]

    # Calculate disparity
    disp = (coord_u_grid - x3) * instance_mask

    return disp
