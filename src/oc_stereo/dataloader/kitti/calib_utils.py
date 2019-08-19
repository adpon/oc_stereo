import csv
import os

import cv2
import numpy as np
import torch


class FrameCalib:
    """Frame Calibration

    Fields:
        p0-p3: (3, 4) Camera P matrices. Contains extrinsic and intrinsic parameters.
        r0_rect: (3, 3) Rectification matrix
        velo_to_cam: (3, 4) Transformation matrix from velodyne to cam coordinate
            Point_Camera = P_cam * R0_rect * Tr_velo_to_cam * Point_Velodyne
        """

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.r0_rect = []
        self.velo_to_cam = []


class StereoCalib:
    """Stereo Calibration

    Fields:
        baseline: distance between the two camera centers
        f: focal length
        k: (3, 3) intrinsic calibration matrix
        p: (3, 4) camera projection matrix
        center_u: camera origin u coordinate
        center_v: camera origin v coordinate
        """

    def __init__(self):
        self.baseline = 0.0
        self.f = 0.0
        self.k = []
        self.center_u = 0.0
        self.center_v = 0.0


def read_frame_calib(calib_file_path):
    """Reads the calibration file for a sample

    Args:
        calib_file_path: calibration file path

    Returns:
        frame_calib: FrameCalib frame calibration
    """

    data_file = open(calib_file_path, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    frame_calib = FrameCalib()
    frame_calib.p0 = p_all[0]
    frame_calib.p1 = p_all[1]
    frame_calib.p2 = p_all[2]
    frame_calib.p3 = p_all[3]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calib.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calib.velo_to_cam = np.reshape(tr_v2c, (3, 4))

    return frame_calib


def get_frame_calib(calib_dir, sample_name):

    calib_file_path = calib_dir + '/{}.txt'.format(sample_name)
    frame_calib = read_frame_calib(calib_file_path)

    return frame_calib


def krt_from_p(p, fsign=1):
    """Factorize the projection matrix P as P=K*[R;t]
    and enforce the sign of the focal length to be fsign.


    Keyword Arguments:
    ------------------
    p : 3x4 list
        Camera Matrix.

    fsign : int
            Sign of the focal length.


    Returns:
    --------
    k : 3x3 list
        Intrinsic calibration matrix.

    r : 3x3 list
        Extrinsic rotation matrix.

    t : 1x3 list
        Extrinsic translation.
    """
    s = p[0:3, 3]
    q = np.linalg.inv(p[0:3, 0:3])
    u, b = np.linalg.qr(q)
    sgn = np.sign(b[2, 2])
    b = b * sgn
    s = s * sgn

    # If the focal length has wrong sign, change it
    # and change rotation matrix accordingly.
    if fsign * b[0, 0] < 0:
        e = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    if fsign * b[2, 2] < 0:
        e = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    # If u is not a rotation matrix, fix it by flipping the sign.
    if np.linalg.det(u) < 0:
        u = -u
        s = -s

    r = np.matrix.transpose(u)
    t = np.matmul(b, s)
    k = np.linalg.inv(b)
    k = k / k[2, 2]

    # Sanity checks to ensure factorization is correct
    if np.linalg.det(r) < 0:
        print('Warning: R is not a rotation matrix.')

    if k[2, 2] < 0:
        print('Warning: K has a wrong sign.')

    return k, r, t


# TODO: check that krt_from_p is needed
def get_stereo_calibration(left_cam_mat, right_cam_mat):
    """Extract parameters required to transform disparity image to 3D point
    cloud.

    Keyword Arguments:
    ------------------
    left_cam_mat : 3x4 list
                   Left Camera Matrix.

    right_cam_mat : 3x4 list
                   Right Camera Matrix.


    Returns:
    --------
    stereo_calibration_info : Instance of StereoCalibrationData class
                              Placeholder for stereo calibration parameters.
    """

    stereo_calib = StereoCalib()
    k_left, r_left, t_left = krt_from_p(left_cam_mat)
    _, _, t_right = krt_from_p(right_cam_mat)

    stereo_calib.baseline = abs(t_left[0] - t_right[0])
    stereo_calib.f = k_left[0, 0]
    stereo_calib.k = k_left
    stereo_calib.center_u = k_left[0, 2]
    stereo_calib.center_v = k_left[1, 2]

    return stereo_calib


def lidar_to_cam_frame(xyz_lidar, frame_calib):
    """Transforms points in lidar frame to the reference camera (cam 0) frame

    Args:
        xyz_lidar: points in lidar frame
        frame_calib: FrameCalib frame calibration

    Returns:
        ret_xyz: (N, 3) points in reference camera (cam 0) frame
    """

    # Pad the r0_rect matrix to a 4x4
    r0_rect_mat = frame_calib.r0_rect
    r0_rect_mat = np.pad(r0_rect_mat, ((0, 1), (0, 1)),
                         'constant', constant_values=0)
    r0_rect_mat[3, 3] = 1

    # Pad the vel_to_cam matrix to a 4x4
    tf_mat = frame_calib.velo_to_cam
    tf_mat = np.pad(tf_mat, ((0, 1), (0, 0)),
                    'constant', constant_values=0)
    tf_mat[3, 3] = 1

    # Pad the point cloud with 1's for the transformation matrix multiplication
    one_pad = np.ones(xyz_lidar.shape[0]).reshape(-1, 1)
    xyz_lidar = np.append(xyz_lidar, one_pad, axis=1)

    # p_cam = P2 * (R0_rect * Tr_velo_to_cam * p_velo)
    rectified = np.dot(r0_rect_mat, tf_mat)
    ret_xyz = np.dot(rectified, xyz_lidar.T)

    # Return (N, 3) points
    return ret_xyz[0:3].T


def pc_from_disparity(disp, stereo_calib, flatten_order='C'):
    """Transforms disparity map to 3d point cloud

    Args:
        disp: disparity map
        stereo_calib: StereoCalib
        flatten_order: (optional) see numpy.ndarray.flatten
            Specifies the way the depth array is flattened
            'C' - (default) row-major (C-style) order
            'F' - column-major (Fortran- style) order

    Returns:
        depth_map: depth map calculated from disparity
    """

    disp = np.asarray(disp, np.float32)
    disp[disp == 0] = 0.1

    depth = np.ones(disp.shape, np.single)
    depth = np.multiply(depth,
                        stereo_calib.f *
                        stereo_calib.baseline)

    depth = np.divide(depth, np.float32(disp))
    depth[depth > 80.0] = 0.0

    sz = np.shape(depth)
    depth = depth.flatten(flatten_order)

    xx, yy = np.meshgrid(
        np.arange(0, sz[1], 1), np.arange(0, sz[0], 1))

    xx = xx.flatten(flatten_order) - stereo_calib.center_u
    yy = yy.flatten(flatten_order) - stereo_calib.center_v

    temp = np.divide(depth, stereo_calib.f)

    x = np.multiply(xx, temp)
    y = np.multiply(yy, temp)
    z = depth

    return x, y, z


def pc_from_disparity_crop(disp, stereo_calib, box_2d, roi_size, flatten_order='C',
                           round_box_2d=True):
    """Transforms disparity map to 3d point cloud

    Args:
        disp: disparity map
        stereo_calib: StereoCalib
        box_2d: 2D box [y1, x1, y2, x2]
        roi_size: [h, w] of region of interest
        flatten_order: (optional) see numpy.ndarray.flatten
            Specifies the way the depth array is flattened
            'C' - (default) row-major (C-style) order
            'F' - column-major (Fortran- style) order
        round_box_2d: (optional) whether to round the 2D box dimensions

    Returns:
        depth_map: depth map calculated from disparity
    """

    if round_box_2d:
        y1, x1, y2, x2 = np.round(box_2d)
    else:
        y1, x1, y2, x2 = box_2d

    num_roi_pixels_x = roi_size[0]
    num_roi_pixels_y = roi_size[1]

    roi_pixel_w = (x2 - x1) / np.float(num_roi_pixels_x)
    roi_pixel_h = (y2 - y1) / np.float(num_roi_pixels_y)

    disp = np.asarray(disp, np.float32)
    disp[disp == 0] = 0.1

    depth = np.ones(disp.shape, np.single)
    depth = np.multiply(depth,
                        stereo_calib.f *
                        stereo_calib.baseline)

    depth = np.divide(depth, np.float32(disp))
    depth[depth > 80.0] = 0.0

    depth = depth.flatten(flatten_order)

    xx, yy = np.meshgrid(
        np.linspace(x1, x2 - roi_pixel_w, num_roi_pixels_x),
        np.linspace(y1, y2 - roi_pixel_h, num_roi_pixels_y))

    xx = xx.flatten(flatten_order) - stereo_calib.center_u
    yy = yy.flatten(flatten_order) - stereo_calib.center_v

    temp = np.divide(depth, stereo_calib.f)

    x = np.multiply(xx, temp)
    y = np.multiply(yy, temp)
    z = depth

    return x, y, z


def torch_pc_from_disparity_crop(disp, stereo_calib_f, stereo_calib_baseline,
                                 stereo_calib_centre_u, stereo_calib_centre_v,
                                 box_2d, roi_size):
    """Transforms disparity map to 3d point cloud

    Args:
        disp: disparity map
        stereo_calib_f: focal length
        stereo_calib_baseline: baseline
        stereo_calib_centre_u:
        stereo_calib_centre_v:
        box_2d:
        roi_size:

    Returns:
        pc: pc calculated from disparity
    """

    y1 = box_2d[:, 0]
    x1 = box_2d[:, 1]
    y2 = box_2d[:, 2]
    x2 = box_2d[:, 3]

    num_roi_pixels_x = roi_size[0]
    num_roi_pixels_y = roi_size[1]

    roi_pixel_w = (x2 - x1) / num_roi_pixels_x
    roi_pixel_h = (y2 - y1) / num_roi_pixels_y

    disp[disp == 0] = 0.1

    depth = stereo_calib_f * stereo_calib_baseline / disp
    depth[depth > 80.0] = 0.0

    sz = depth.shape
    batch_size = sz[0]

    all_xx = []
    all_yy = []
    for idx in range(batch_size):
        yy, xx = torch.meshgrid(
            torch.linspace(y1[idx], y2[idx] - roi_pixel_w[idx], num_roi_pixels_x.type(
                torch.int32)).cuda(),
            torch.linspace(x1[idx], x2[idx] - roi_pixel_h[idx], num_roi_pixels_y.type(
                torch.int32)).cuda())
        all_xx.append(xx)
        all_yy.append(yy)

    all_xx = torch.stack(all_xx)
    all_yy = torch.stack(all_yy)

    i = all_xx - stereo_calib_centre_u
    j = all_yy - stereo_calib_centre_v

    ratio = depth / stereo_calib_f

    x = i * ratio
    y = j * ratio
    z = depth

    xyz = torch.stack([x, y, z]).transpose(1, 0)
    pc = xyz.view(batch_size, 3, -1)

    return pc


def torch_pc_from_disparity(disp, stereo_calib_f, stereo_calib_baseline,
                            stereo_calib_centre_u, stereo_calib_centre_v,
                            flatten_order='C'):
    """Transforms disparity map to 3d point cloud

    Args:
        disp: disparity map
        stereo_calib_f: focal length
        stereo_calib_baseline: baseline
        flatten_order: (optional) see numpy.ndarray.flatten
            Specifies the way the depth array is flattened
            'C' - (default) row-major (C-style) order
            'F' - column-major (Fortran- style) order

    Returns:
        depth_map: depth map calculated from disparity
    """

    disp[disp == 0] = 0.1

    depth = stereo_calib_f * stereo_calib_baseline / disp
    depth[depth > 80.0] = 0.0

    sz = depth.shape
    batch_size = sz[0]
    depth_height = sz[1]
    depth_width = sz[2]

    xx, yy = torch.meshgrid(
        torch.arange(0, depth_width, 1), torch.arange(0, depth_height, 1))

    xx = xx.type(torch.float32).cuda()
    yy = yy.type(torch.float32).cuda()

    i = torch.transpose(xx, 0, 1) - stereo_calib_centre_u
    j = torch.transpose(yy, 0, 1) - stereo_calib_centre_v

    ratio = depth / stereo_calib_f

    x = i * ratio
    y = j * ratio
    z = depth

    xyz = torch.transpose(torch.stack([x, y, z]), 1, 0)
    # pc = xyz.view(batch_size, 3, -1)

    return xyz


def project_pc_to_image(point_cloud, cam_p):
    """Projects a 3D point cloud to 2D points

    Args:
        point_cloud: (3, N) point cloud
        cam_p: camera projection matrix

    Returns:
        pts_2d: (2, N) projected coordinates [u, v] of the 3D points
    """

    pc_padded = np.append(point_cloud, np.ones((1, point_cloud.shape[1])), axis=0)
    pts_2d = np.dot(cam_p, pc_padded)

    pts_2d[0:2] = pts_2d[0:2] / pts_2d[2]
    return pts_2d[0:2]


def read_disparity(disp_dir, img_idx):
    """Reads in Disparity file from Kitti Dataset.

        Keyword Arguments:
        ------------------
        calib_dir : Str
                    Directory of the disparity files.

        img_idx : Int
                  Index of the image.

        Returns:
        --------
        disp_img : Numpy Array
                   Contains the disparity image.

        [] : if file is not found

        """
    disp_path = disp_dir + "/%06d_left_disparity.png" % img_idx

    if os.path.exists(disp_path):
        disp_img = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH)
        return disp_img
    else:
        raise ValueError('Disparity map not found')
