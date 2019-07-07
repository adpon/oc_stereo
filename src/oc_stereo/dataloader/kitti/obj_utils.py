import copy
import os

import cv2
import numpy as np

from oc_stereo.builders.config_builder_util import ConfigObj

from oc_stereo.dataloader.kitti import calib_utils, depth_map_utils, instance_utils, evaluation

# KITTI difficulty thresholds (easy, moderate, hard)
HEIGHT = (40, 25, 25)
OCCLUSION = (0, 1, 2)
TRUNCATION = (0.15, 0.3, 0.5)

# Mean object heights from hist_labels.py
MEAN_HEIGHTS = {
    'Car': 1.526,
    'Pedestrian': 1.761,
    'Cyclist': 1.737,
}


class Difficulty:

    # Values as integers for indexing
    EASY = 0
    MODERATE = 1
    HARD = 2
    ALL = 3

    # Mappings from difficulty to string
    DIFF_TO_STR_MAPPING = {
        EASY: 'easy',
        MODERATE: 'moderate',
        HARD: 'hard',
        ALL: 'all',
    }

    # Mappings from strings to difficulty
    STR_TO_DIFF_MAPPING = {
        'easy': EASY,
        'moderate': MODERATE,
        'hard': HARD,
        'all': ALL,
    }

    @staticmethod
    def to_string(difficulty):
        return Difficulty.DIFF_TO_STR_MAPPING[difficulty]

    @staticmethod
    def from_string(difficulty_str):
        return Difficulty.STR_TO_DIFF_MAPPING[difficulty_str]


class ObjectFilter:
    def __init__(self, config):
        self.classes = config.classes
        self.difficulty = Difficulty.from_string(config.difficulty_str)
        self.box_2d_height = config.box_2d_height
        self.truncation = config.truncation
        self.occlusion = config.occlusion
        self.depth_range = config.depth_range

    @staticmethod
    def create_obj_filter(classes,
                          difficulty,
                          occlusion,
                          truncation,
                          box_2d_height,
                          depth_range):
        config = ConfigObj()
        config.classes = classes
        config.difficulty_str = Difficulty.to_string(difficulty)
        config.occlusion = occlusion
        config.truncation = truncation
        config.box_2d_height = box_2d_height
        config.depth_range = depth_range

        return ObjectFilter(config)


class ObjectLabel:
    """Object Label

    Fields:
        type (str): Object type, one of
            'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
            'Cyclist', 'Tram', 'Misc', or 'DontCare'
        truncation (float): Truncation level, float from 0 (non-truncated) to 1 (truncated),
            where truncated refers to the object leaving image boundaries
        occlusion (int): Occlusion level,  indicating occlusion state:
            0 = fully visible,
            1 = partly occluded,
            2 = largely occluded,
            3 = unknown
        alpha (float): Observation angle of object [-pi, pi]
        x1, y1, x2, y2 (float): 2D bounding box of object in the image. (top left, bottom right)
        h, w, l: 3D object dimensions: height, width, length (in meters)
        t: 3D object centroid x, y, z in camera coordinates (in meters)
        ry: Rotation around Y-axis in camera coordinates [-pi, pi]
        score: Only for results, indicating confidence in detection, needed for p/r curves.
    """

    def __init__(self):
        self.type = None  # Type of object
        self.truncation = 0.0
        self.occlusion = 0
        self.alpha = 0.0
        self.x1 = 0.0
        self.y1 = 0.0
        self.x2 = 0.0
        self.y2 = 0.0
        self.h = 0.0
        self.w = 0.0
        self.l = 0.0
        self.t = (0.0, 0.0, 0.0)
        self.ry = 0.0
        self.score = 0.0

    def __eq__(self, other):
        """Compares the given object to the current ObjectLabel instance.

        :param other: object to compare to this instance against
        :return: True, if other and current instance is the same
        """
        if not isinstance(other, ObjectLabel):
            return False

        if self.__dict__ != other.__dict__:
            return False
        else:
            return True

    def __repr__(self):

        return '({}, a:{}, t:{} lwh:({:.03f}, {:.03f}, {:.03f}), ry:{:.03f})'.format(
            self.type, self.alpha, self.t, self.l, self.w, self.h, self.ry)


def read_labels(label_dir, sample_name):
    """Reads in label data file from Kitti Dataset
    Args:
        label_dir: label directory
        sample_name: sample_name
    Returns:
        obj_list: list of ObjectLabels
    """

    # Check label file
    label_path = label_dir + '/{}.txt'.format(sample_name)

    if not os.path.exists(label_path):
        raise ValueError('Label file could not be found:', label_path)
    if os.stat(label_path).st_size == 0:
        return []

    labels = np.loadtxt(label_path, delimiter=' ', dtype=str, ndmin=2)
    num_rows, num_cols = labels.shape
    if num_cols not in [15, 16]:
        raise ValueError('Invalid label format')

    num_labels = num_rows
    is_results = num_cols == 16

    obj_list = []
    for obj_idx in np.arange(num_labels):
        obj = ObjectLabel()

        # Fill in the object list
        obj.type = labels[obj_idx, 0]
        obj.truncation = float(labels[obj_idx, 1])
        obj.occlusion = float(labels[obj_idx, 2])
        obj.alpha = float(labels[obj_idx, 3])

        obj.x1, obj.y1, obj.x2, obj.y2 = (labels[obj_idx, 4:8]).astype(np.float32)
        obj.h, obj.w, obj.l = (labels[obj_idx, 8:11]).astype(np.float32)
        obj.t = (labels[obj_idx, 11:14]).astype(np.float32)
        obj.ry = float(labels[obj_idx, 14])

        if is_results:
            obj.score = float(labels[obj_idx, 15])
        else:
            obj.score = 0.0

        obj_list.append(obj)

    return np.asarray(obj_list)


def filter_labels_by_class(obj_labels, classes):
    """Filters object labels by classes.

    Args:
        obj_labels: List of object labels
        classes: List of classes to keep, e.g. ['Car', 'Pedestrian', 'Cyclist']

    Returns:
        obj_labels: List of filtered labels
        class_mask: Mask of labels to keep
    """
    class_mask = [(obj.type in classes) for obj in obj_labels]
    return obj_labels[class_mask], class_mask


def filter_labels_by_difficulty(obj_labels, difficulty):
    """Filters object labels by difficulty.

    Args:
        obj_labels: List of object labels
        difficulty: Difficulty level

    Returns:
        obj_labels: List of filtered labels
        difficulty_mask: Mask of labels to keep
    """
    difficulty_mask = [_check_difficulty(obj, difficulty) for obj in obj_labels]
    return obj_labels[difficulty_mask], difficulty_mask


def _check_difficulty(obj, difficulty):
    if difficulty == Difficulty.ALL:
        return True

    return ((obj.occlusion <= OCCLUSION[difficulty]) and
            (obj.truncation <= TRUNCATION[difficulty]) and
            (obj.y2 - obj.y1) >= HEIGHT[difficulty])


def filter_labels_by_box_2d_height(obj_labels, box_2d_height):
    """Filters object labels by 2D box height.

    Args:
        obj_labels: List of object labels
        box_2d_height: Minimum 2D box height

    Returns:
        obj_labels: List of filtered labels
        height_mask: Mask of labels to keep
    """
    height_mask = [(obj_label.y2 - obj_label.y1) > box_2d_height
                   for obj_label in obj_labels]

    return obj_labels[height_mask], height_mask


def filter_labels_by_truncation(obj_labels, truncation):
    """Filters object labels by truncation.

    Args:
        obj_labels: List of object labels
        truncation: Maximum truncation

    Returns:
        obj_labels: List of filtered labels
        trunc_mask: Mask of labels to keep
    """
    trunc_mask = [obj_label.truncation < truncation
                  for obj_label in obj_labels]

    return obj_labels[trunc_mask], trunc_mask


def filter_labels_by_occlusion(obj_labels, occlusion):
    """Filters object labels by truncation.

    Args:
        obj_labels: List of object labels
        occlusion: Maximum occlusion

    Returns:
        obj_labels: List of filtered labels
        occ_mask: Mask of labels to keep
    """
    occ_mask = [obj_label.occlusion < occlusion
                for obj_label in obj_labels]

    return obj_labels[occ_mask], occ_mask


def filter_labels_by_depth_range(obj_labels, depth_range):
    """Filters object labels within a range of depth values.

    Args:
        obj_labels: List of object labels
        depth_range: Range of depth to keep objects

    Returns:
        obj_labels: List of filtered labels
        depth_mask: Mask of labels to keep
    """

    depth_mask = [depth_range[0] < obj_label.t[2] < depth_range[1]
                  for obj_label in obj_labels]

    return obj_labels[depth_mask], depth_mask


def filter_labels(obj_labels, classes=None, difficulty=None,
                  box_2d_height=None, occlusion=None, truncation=None, depth_range=None):
    """Filters object labels by various metrics.

    Args:
        obj_labels: List of object labels
        classes: List of classes to keep, e.g. ['Car', 'Pedestrian', 'Cyclist']
        difficulty: Difficulty level
        box_2d_height: Minimum 2D box height
        occlusion: Minimum occlusion level
        truncation: Minimum truncation level
        depth_range: Range of depth to keep objects

    Returns:
        obj_labels: List of filtered labels
        obj_mask: Mask of labels to keep
    """

    obj_mask = np.full(len(obj_labels), True)

    if classes is not None:
        _, class_mask = filter_labels_by_class(obj_labels, classes)
        obj_mask &= class_mask

    if difficulty is not None:
        _, difficulty_mask = filter_labels_by_difficulty(obj_labels, difficulty)
        obj_mask &= difficulty_mask

    if box_2d_height is not None:
        _, height_mask = filter_labels_by_box_2d_height(obj_labels, box_2d_height)
        obj_mask &= height_mask

    if occlusion is not None:
        _, occ_mask = filter_labels_by_occlusion(obj_labels, occlusion)
        obj_mask &= occ_mask

    if truncation is not None:
        _, trunc_mask = filter_labels_by_truncation(obj_labels, truncation)
        obj_mask &= trunc_mask

    if depth_range is not None:
        _, depth_mask = filter_labels_by_depth_range(obj_labels, depth_range)
        obj_mask &= depth_mask

    return obj_labels[obj_mask], obj_mask


def apply_obj_filter(obj_labels, obj_filter):
    """Applies an ObjectFilter to a list of labels

    Args:
        obj_labels:
        obj_filter (ObjectFilter):

    Returns:
        obj_labels: List of filtered labels
        obj_mask: Mask of labels to keep
    """
    obj_labels, obj_mask = filter_labels(
        obj_labels,
        classes=obj_filter.classes,
        difficulty=obj_filter.difficulty,
        box_2d_height=obj_filter.box_2d_height,
        occlusion=obj_filter.occlusion,
        truncation=obj_filter.truncation,
        depth_range=obj_filter.depth_range)

    return obj_labels, obj_mask


def boxes_2d_from_obj_labels(obj_labels):
    return np.asarray([object_label_to_box_2d(obj_label)
                       for obj_label in obj_labels], np.float32)


def boxes_3d_from_obj_labels(obj_labels):
    return np.asarray([object_label_to_box_3d(obj_label)
                       for obj_label in obj_labels], np.float32)


def obj_classes_from_obj_labels(obj_labels):
    return np.asarray([obj_label.type
                       for obj_label in obj_labels])


def get_image(sample_name, image_dir):
    image_path = image_dir + '/{}.png'.format(sample_name)
    image = cv2.imread(image_path)
    return image


def get_instance_masks(sample_name, instance_dir, num_objs):
    """Gets the instance masks

    Args:
        sample_name: Sample name
        instance_dir: Instance directory
        num_objs: Total number of objects in the scene

    Returns:
        instance_masks: (N, H, W) Instance masks
    """
    instance_img = instance_utils.get_instance_image(sample_name, instance_dir)
    instance_masks = instance_utils.get_instance_mask_list(instance_img, num_objs)
    return instance_masks


def read_lidar(velo_dir, sample_name):
    """Reads the lidar bin file for a sample

    Args:
        velo_dir: velodyne directory
        sample_name: sample name

    Returns:
        xyzi: (N, 4) points and intensities
    """

    velo_path = velo_dir + '/{}.bin'.format(sample_name)

    if os.path.exists(velo_path):
        with open(velo_path, 'rb') as fid:
            data_array = np.fromfile(fid, np.single)

        xyzi = data_array.reshape(-1, 4)

        return xyzi
    else:
        raise ValueError('Velodyne file not found')


def get_lidar_point_cloud(sample_name, frame_calib, velo_dir):
    """Gets the lidar point cloud in cam0 frame.

    Args:
        sample_name: Sample name
        frame_calib: FrameCalib
        velo_dir: Velodyne directory

    Returns:
        (3, N) point_cloud in the form [[x,...][y,...][z,...]]
    """

    xyzi = read_lidar(velo_dir, sample_name)

    # Calculate the point cloud
    points_in_lidar_frame = xyzi[:, 0:3]
    points = calib_utils.lidar_to_cam_frame(points_in_lidar_frame, frame_calib)

    return points.T


def get_lidar_point_cloud_for_cam(sample_name, frame_calib, velo_dir,
                                  image_shape=None, cam_idx=2):
    """Gets the lidar point cloud in cam0 frame, and optionally returns only the
    points that are projected to an image.

    Args:
        sample_name: sample name
        frame_calib: FrameCalib frame calibration
        velo_dir: velodyne directory
        image_shape: (optional) image shape [h, w] to filter points inside image
        cam_idx: (optional) cam index (2 or 3) for filtering

    Returns:
        (3, N) point_cloud in the form [[x,...][y,...][z,...]]
    """

    # Get points in camera frame
    point_cloud = get_lidar_point_cloud(sample_name, frame_calib, velo_dir)

    # Only keep points in front of camera (positive z)
    point_cloud = point_cloud[:, point_cloud[2] > 1.0]

    if image_shape is None:
        return point_cloud

    else:

        # Project to image frame
        if cam_idx == 2:
            cam_p = frame_calib.p2
        elif cam_idx == 3:
            cam_p = frame_calib.p3
        else:
            raise ValueError('Invalid cam_idx', cam_idx)

        # Project to image
        points_in_img = calib_utils.project_pc_to_image(point_cloud, cam_p=cam_p)
        points_in_img_rounded = np.round(points_in_img)

        # Filter based on the given image shape
        image_filter = (points_in_img_rounded[0] >= 0) & \
                       (points_in_img_rounded[0] < image_shape[1]) & \
                       (points_in_img_rounded[1] >= 0) & \
                       (points_in_img_rounded[1] < image_shape[0])

        filtered_point_cloud = point_cloud[:, image_filter].astype(np.float32)

        return filtered_point_cloud


def get_stereo_point_cloud(sample_name, calib_dir, disp_dir):
    """
    Gets the point cloud for an image calculated from the disparity map

    :param sample_name: sample name
    :param calib_dir: directory with calibration files
    :param disp_dir: directory with disparity images

    :return: (3, N) point_cloud in the form [[x,...][y,...][z,...]]
    """

    # Read calibration info
    frame_calib = calib_utils.get_frame_calib(calib_dir, sample_name)
    stereo_calibration_info = calib_utils.get_stereo_calibration(frame_calib.p2,
                                                                 frame_calib.p3)

    # Read disparity
    disp = cv2.imread(disp_dir + '/{}.png'.format(sample_name),
                      cv2.IMREAD_ANYDEPTH)
    disp = np.float32(disp)
    disp = np.divide(disp, 256)
    disp[disp == 0] = 0.1

    # Calculate the point cloud
    point_cloud = calib_utils.depth_from_disparity(disp, stereo_calibration_info)

    return point_cloud


def get_depth_map_path(sample_name, depth_dir):
    return depth_dir + '/{}.png'.format(sample_name)


def get_disp_map_path(sample_name, disp_dir):
    return disp_dir + '/{}.png'.format(sample_name)


def get_disp_map(sample_name, disp_dir):
    disp_map_path = get_disp_map_path(sample_name, disp_dir)
    disp_map = cv2.imread(disp_map_path, cv2.IMREAD_ANYDEPTH)
    disp_map = disp_map / 256.0
    return disp_map


def get_depth_map(sample_name, depth_dir):
    depth_map_path = get_depth_map_path(sample_name, depth_dir)
    depth_map = depth_map_utils.read_depth_map(depth_map_path)
    return depth_map


def get_depth_map_point_cloud(sample_name, frame_calib, depth_dir):
    """Calculates the point cloud from a depth map

    Args:
        sample_name: sample name
        frame_calib: FrameCalib frame calibration
        depth_dir: folder with depth maps
        cam_idx: cam index (2 or 3)

    Returns:
        (3, N) point_cloud in the form [[x,...][y,...][z,...]]
    """
    depth_map_path = get_depth_map_path(sample_name, depth_dir)
    depth_map = depth_map_utils.read_depth_map(depth_map_path)
    depth_map_shape = depth_map.shape[0:2]

    # Calculate point cloud from depth map
    cam_p = frame_calib.p2
    # cam_p = frame_calib.p2 if cam_idx == 2 else frame_calib.p3

    # # TODO: Call depth_map_utils version
    return depth_map_utils.get_depth_point_cloud(depth_map, cam_p)

    # Calculate points from depth map
    depth_map_flattened = depth_map.flatten()
    xx, yy = np.meshgrid(np.arange(0, depth_map_shape[1], 1),
                         np.arange(0, depth_map_shape[0], 1))
    xx = xx.flatten() - cam_p[0, 2]
    yy = yy.flatten() - cam_p[1, 2]

    temp = np.divide(depth_map_flattened, cam_p[0, 0])
    x = np.multiply(xx, temp)
    y = np.multiply(yy, temp)
    z = depth_map_flattened

    # Get x offset (b_cam) from calibration: cam_mat[0, 3] = (-f_x * b_cam)
    x_offset = -cam_p[0, 3] / cam_p[0, 0]

    point_cloud = np.asarray([x + x_offset, y, z])

    return point_cloud.astype(np.float32)


def get_road_plane(sample_name, planes_dir):
    """Reads the ground plane from file

    Args:
        sample_name: sample name
        planes_dir: planes directory

    Returns:
        normalized plane coefficients [a, b, c, d] satisfying ax+by+cz+d=0
    """

    plane_file = planes_dir + '/{}.txt'.format(sample_name)

    with open(plane_file, 'r') as input_file:
        lines = input_file.readlines()
        input_file.close()

    # Plane coefficients stored in 4th row
    lines = lines[3].split()

    # Convert str to float
    lines = [float(i) for i in lines]

    plane = np.asarray(lines)

    # Ensure normal is always facing up.
    # In Kitti's frame of reference, +y is down
    if plane[1] > 0:
        plane = -plane
        raise ValueError('Plane is facing downwards')

    # Normalize the plane coefficients
    norm = np.linalg.norm(plane[0:3])
    plane = plane / norm

    return plane


def compute_obj_label_corners_3d(object_label):
    """
    Computes the 3D bounding box corner positions from an ObjectLabel

    :param object_label: ObjectLabel to compute corners from
    :return: a numpy array of 3D corners if the box is in front of the camera,
             an empty array otherwise
    """

    # compute rotational matrix
    rot = np.array([[+np.cos(object_label.ry), 0, +np.sin(object_label.ry)],
                    [0, 1, 0],
                    [-np.sin(object_label.ry), 0, +np.cos(object_label.ry)]])

    l = object_label.l
    w = object_label.w
    h = object_label.h

    # 3D BB corners
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))

    corners_3d[0, :] = corners_3d[0, :] + object_label.t[0]
    corners_3d[1, :] = corners_3d[1, :] + object_label.t[1]
    corners_3d[2, :] = corners_3d[2, :] + object_label.t[2]

    return corners_3d


# TODO: Move to box_3d_projector?
def project_corners_3d_to_image(corners_3d, p):
    """Computes the 3D bounding box projected onto
    image space.

    Keyword arguments:
    obj -- object file to draw bounding box
    p -- transform matrix

    Returns:
        corners : numpy array of corner points projected
        onto image space.
        face_idx: numpy array of 3D bounding box face
    """
    # index for 3d bounding box face
    # it is converted to 4x4 matrix
    face_idx = np.array([0, 1, 5, 4,  # front face
                         1, 2, 6, 5,  # right face
                         2, 3, 7, 6,  # back face
                         3, 0, 4, 7]).reshape((4, 4))  # left face
    return calib_utils.project_pc_to_image(corners_3d, p), face_idx


def points_in_img_filter(points_in_img, image_shape):
    # Filter based on the given image size
    image_filter = (points_in_img[0] >= 0) & (points_in_img[0] < image_shape[1]) & \
                   (points_in_img[1] >= 0) & (points_in_img[1] < image_shape[0])

    return image_filter


def filter_pc_to_image(point_cloud, points_in_img, image_shape):
    image_filter = points_in_img_filter(points_in_img, image_shape)

    return point_cloud[:, image_filter], image_filter


def is_point_inside(points, box_corners):
    """Check if each point in a 3D point cloud lies within the 3D bounding box

    If we think of the bounding box as having bottom face
    defined by [P1, P2, P3, P4] and top face [P5, P6, P7, P8]
    then there are three directions on a perpendicular edge:
        u = P1 - P2
        v = P1 - P4
        w = P1 - P5

    A point x lies within the box when the following constraints
    are respected:
        - The dot product u.x is between u.P1 and u.P2
        - The dot product v.x is between v.P1 and v.P4
        - The dot product w.x is between w.P1 and w.P5

    :param points: 3D point cloud to test in the form [[x1...xn],[y1...yn],[z1...zn]]
    :param box_corners: 3D corners of the bounding box

    :return bool mask of which points are within the bounding box.
            Use numpy function .all() to check all points
    """

    p1 = box_corners[:, 0]
    p2 = box_corners[:, 1]
    p4 = box_corners[:, 3]
    p5 = box_corners[:, 4]

    u = p2 - p1
    v = p4 - p1
    w = p5 - p1

    # if u.P1 < u.x < u.P2
    u_dot_x = np.dot(u, points)
    u_dot_p1 = np.dot(u, p1)
    u_dot_p2 = np.dot(u, p2)

    # if v.P1 < v.x < v.P4
    v_dot_x = np.dot(v, points)
    v_dot_p1 = np.dot(v, p1)
    v_dot_p2 = np.dot(v, p4)

    # if w.P1 < w.x < w.P5
    w_dot_x = np.dot(w, points)
    w_dot_p1 = np.dot(w, p1)
    w_dot_p2 = np.dot(w, p5)

    point_mask = (u_dot_p1 < u_dot_x) & (u_dot_x < u_dot_p2) & \
                 (v_dot_p1 < v_dot_x) & (v_dot_x < v_dot_p2) & \
                 (w_dot_p1 < w_dot_x) & (w_dot_x < w_dot_p2)

    return point_mask


def get_area_filter(point_cloud, extents):
    """

    Args:
        point_cloud: (3, N) point cloud
        extents: 3D area in the form [[min_x, max_x], [min_y, max_y], [min_z, max_z]]

    Returns:

    """
    if not isinstance(point_cloud, np.ndarray) and isinstance(extents, np.ndarray):
        raise TypeError('point_cloud and extents must be of type np.ndarray')

    extents_filter = \
        (point_cloud[0] > extents[0, 0]) & \
        (point_cloud[0] < extents[0, 1]) & \
        (point_cloud[1] > extents[1, 0]) & \
        (point_cloud[1] < extents[1, 1]) & \
        (point_cloud[2] > extents[2, 0]) & \
        (point_cloud[2] < extents[2, 1])

    return extents_filter


def filter_pc_to_area(point_cloud, area_extents):

    area_filter = get_area_filter(point_cloud, area_extents)

    return point_cloud[:, area_filter], area_filter


def compute_box_3d_corners(box_3d):
    """Computes box corners

    :param box_3d: input 3D boxes 1 x 7 array represented as [x, y, z, l, w, h, ry]

    :return: corners_3d: array of box corners 8 x [x y z]
    """

    tx, ty, tz, l, w, h, ry = box_3d

    half_l = l / 2
    half_w = w / 2

    # Compute rotation matrix
    rot = np.array([[+np.cos(ry), 0, +np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, +np.cos(ry)]])

    # 3D BB corners
    x_corners = np.array(
        [half_l, half_l, -half_l, -half_l, half_l, half_l, -half_l, -half_l])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array(
        [half_w, -half_w, -half_w, half_w, half_w, -half_w, -half_w, half_w])
    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + tx
    corners_3d[1, :] = corners_3d[1, :] + ty
    corners_3d[2, :] = corners_3d[2, :] + tz

    return np.array(corners_3d)


def points_in_box_3d(box_3d, points):
    """Finds the points inside a box_3d.

    Args:
        box_3d: box_3d
        points: (N, 3) points

    Returns:
        points in box_3d
        mask of points
    """

    # Find box corners
    corners_3d = compute_box_3d_corners(box_3d).T

    # Compute box's principle axes
    u = corners_3d[0, :] - corners_3d[1, :]
    v = corners_3d[0, :] - corners_3d[3, :]
    w = corners_3d[0, :] - corners_3d[4, :]

    # Compute test bounds for point to be included inside bounding box
    up0 = np.dot(u, corners_3d[0, :])
    up1 = np.dot(u, corners_3d[1, :])

    vp0 = np.dot(v, corners_3d[0, :])
    vp3 = np.dot(v, corners_3d[3, :])

    wp0 = np.dot(w, corners_3d[0, :])
    wp4 = np.dot(w, corners_3d[4, :])

    # Compute dot product between point cloud and principle axes
    u_dot = np.matmul(points, u)
    v_dot = np.matmul(points, v)
    w_dot = np.matmul(points, w)

    # Create mask from bounds and return points inside bounding box
    mask_u = np.logical_and(u_dot <= up0, u_dot >= up1)
    mask_v = np.logical_and(v_dot <= vp0, v_dot >= vp3)
    mask_w = np.logical_and(w_dot <= wp0, w_dot >= wp4)

    mask = np.logical_and(mask_u, mask_v)
    mask = np.logical_and(mask, mask_w)

    return points[mask], mask


def get_viewing_angle_box_2d(box_2d, cam_p):
    """Estimates the viewing angle towards an object given the 2D box and
    camera projection matrix.

    Args:
        box_2d: 2D box [y1, x1, y2, x2]
        cam_p: camera projection matrix

    Returns:
        viewing_angle: viewing angle to object
    """
    # Find centre of box
    centre_x = np.mean(box_2d[[1, 3]])

    centre_u = cam_p[0, 2]
    focal_length = cam_p[0, 0]

    # Assume depth of 1.0 to calculate viewing angle
    # viewing_angle = atan2(i / f, 1.0)
    viewing_angle = np.arctan2((centre_x - centre_u) / focal_length, 1.0)

    return viewing_angle


def get_viewing_angle_box_3d(box_3d, cam_p=None, version='x_offset'):
    """Calculates the viewing angle to the centroid of a box_3d

    Args:
        box_3d: box_3d in cam_0 frame
        cam_p: camera projection matrix, required if version is not 'cam_0'
        version:
            'cam_0': assuming cam_0 frame
            'x_offset': apply x_offset from camera baseline to cam_0
            'projection': project centroid to image

    Returns:
        viewing_angle: viewing angle to box centroid
    """
    check_box_3d_format(box_3d)

    if version == 'cam_0':
        # Viewing angle in cam_0 frame
        viewing_angle = np.arctan2(box_3d[0], box_3d[2])

    elif version == 'x_offset':
        # Get x offset (b_cam) from calibration: cam_mat[0, 3] = (-f_x * b_cam)
        x_offset = -cam_p[0, 3] / cam_p[0, 0]

        # Shift box_3d from cam_0 to cam_N frame
        box_x_cam = box_3d[0] - x_offset

        # Calculate viewing angle
        viewing_angle = np.arctan2(box_x_cam, box_3d[2])

    elif version == 'projection':
        # Project centroid to the image
        proj_uv = calib_utils.project_pc_to_image(box_3d[0:3].reshape(3, -1), cam_p)

        centre_u = cam_p[0, 2]
        focal_length = cam_p[0, 0]

        centre_x = proj_uv[0][0]

        # Assume depth of 1.0 to calculate viewing angle
        # viewing_angle = atan2(i / f, 1.0)
        viewing_angle = np.arctan2((centre_x - centre_u) / focal_length, 1.0)

    else:
        raise ValueError('Invalid version', version)

    return viewing_angle


def obj_label_to_kitti_fmt(obj_label):
    """Converts an object label to kitti format
    [type, truncation, occlusion, alpha(observation angle),
    (x1, y1, x2, y2),
    (h, w, l),
    (x, y, z), ry,
    score]
    """

    return [
        obj_label.type, -1, -1, obj_label.alpha,
        obj_label.x1, obj_label.y1, obj_label.x2, obj_label.y2,
        obj_label.h, obj_label.w, obj_label.l,
        obj_label.t[0], obj_label.t[1], obj_label.t[2], obj_label.ry,
        obj_label.score
    ]


def check_obj_label_format(input_data):
    """Checks for correct ObjectLabel format."""
    if not isinstance(input_data, ObjectLabel):
        raise TypeError('Given input is not an ObjectLabel.')


def check_box_3d_format(input_data):
    """Checks for correct box_3d format. If not proper type, raises error.

    Args:
        input_data: input numpy array or tensor to check for valid box_3d format
    """

    # Check type
    if isinstance(input_data, np.ndarray):
        # Check for size for numpy array form (N x 7)
        if input_data.ndim == 2:
            if input_data.shape[1] != 7:
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be N x 7 for box_3d.')
        elif input_data.ndim == 1:
            if input_data.shape[0] != 7:
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be 7 for box_3d.')
    else:
        raise TypeError('Given input is not of valid types.'
                        '(i.e. np.ndarray or tf.Tensor)')


def box_3d_to_object_label(box_3d, obj_type='Car'):
    """Turns a box_3d into an ObjectLabel

    Args:
        box_3d: 3D box in the format [x, y, z, l, w, h, ry]
        obj_type: Optional, the object type

    Returns:
        ObjectLabel with the location, size, and rotation filled out
    """

    check_box_3d_format(box_3d)

    obj_label = ObjectLabel()

    obj_label.type = obj_type

    obj_label.t = box_3d.take((0, 1, 2))
    obj_label.l = box_3d[3]
    obj_label.w = box_3d[4]
    obj_label.h = box_3d[5]
    obj_label.ry = box_3d[6]

    return obj_label


def object_label_to_box_2d(obj_label):
    """Converts an ObjectLabel into box_2d

    Args:
        obj_label: ObjectLabel to convert

    Returns:
        box_2d: 2D box in box_2d format [y1, x1, y2, x2]
    """
    check_obj_label_format(obj_label)
    box_2d = np.asarray([obj_label.y1, obj_label.x1, obj_label.y2, obj_label.x2], np.float32)
    return box_2d


def object_label_to_box_3d(obj_label):
    """Turns an ObjectLabel into a box_3d

    Args:
        obj_label: ObjectLabel

    Returns:
        box_3d: 3D box in box_3d format [x, y, z, l, w, h, ry]
    """

    check_obj_label_format(obj_label)

    box_3d = np.zeros(7, dtype=np.float32)

    box_3d[0:3] = obj_label.t
    box_3d[3:6] = obj_label.l, obj_label.w, obj_label.h
    box_3d[6] = obj_label.ry

    return box_3d


def boxes_2d_to_iou_fmt(boxes_2d):
    """Converts a list of boxes_2d [y1, x1, y2, x2] to
    iou format [x1, y1, x2, y2]"""
    boxes_2d = np.asarray(boxes_2d)
    boxes_2d_iou_fmt = boxes_2d[:, [1, 0, 3, 2]]
    return boxes_2d_iou_fmt


# TODO Remove dependency on this function
def box_3d_to_3d_iou_format(boxes_3d):
    """ Returns a numpy array of 3d box format for iou calculation
    Args:
        boxes_3d: list of 3d boxes
    Returns:
        new_anchor_list: numpy array of 3d box format for iou
    """
    boxes_3d = np.asarray(boxes_3d)
    check_box_3d_format(boxes_3d)

    iou_3d_boxes = np.zeros([len(boxes_3d), 7])
    iou_3d_boxes[:, 4:7] = boxes_3d[:, 0:3]
    iou_3d_boxes[:, 1] = boxes_3d[:, 3]
    iou_3d_boxes[:, 2] = boxes_3d[:, 4]
    iou_3d_boxes[:, 3] = boxes_3d[:, 5]
    iou_3d_boxes[:, 0] = boxes_3d[:, 6]

    return iou_3d_boxes
