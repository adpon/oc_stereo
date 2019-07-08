import copy

import cv2
import numpy as np

from oc_stereo.core import transform_utils
from oc_stereo.dataloader.kitti import calib_utils, depth_map_utils


def read_instance_image(instance_image_path):
    instance_image = cv2.imread(instance_image_path, cv2.IMREAD_GRAYSCALE)
    return instance_image


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

        all_instance_maps.append(instance_map_resized)

    return np.asarray(all_instance_maps), np.asarray(all_valid_mask_maps)


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


def make_boxes_same_height(boxes_left, boxes_right):
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

    i3_prime = (i2_prime - local_disp) * instance_mask

    # Box width
    box_width = box_right[3] - box_right[1]

    i3 = i3_prime / roi_size[1] * box_width

    x3 = i3 + box_right[1]

    # Calculate disparity
    disp = (coord_u_grid - x3) * instance_mask

    return disp
