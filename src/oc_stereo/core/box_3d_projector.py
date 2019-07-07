import numpy as np

from oc_stereo.dataloader.kitti import calib_utils, obj_utils


def project_to_image_space(box_3d, calib_p2,
                           truncate=False, image_shape=None,
                           discard=True, discard_before_truncation=True):
    """ Projects a box_3d into image space

    Args:
        box_3d: single box_3d to project
        calib_p2: stereo calibration p2 matrix
        truncate: if True, 2D projections are truncated to be inside the image
        image_shape: [h, w] must be provided if truncate is True,
            used for truncation
        discard: if True, discard boxes that are truncated over a certain amount
        discard_before_truncation: If True, discard boxes that are larger than
            80% of the image in width OR height BEFORE truncation. If False,
            discard boxes that are larger than 80% of the width AND
            height AFTER truncation.

    Returns:
        Projected box in image space [y1, x1, y2, x2]
            Returns None if box is not inside the image
    """

    obj_utils.check_box_3d_format(box_3d)

    obj_label = obj_utils.box_3d_to_object_label(box_3d)
    corners_3d = obj_utils.compute_obj_label_corners_3d(obj_label)

    projected = calib_utils.project_pc_to_image(corners_3d, calib_p2)

    x1 = np.amin(projected[0])
    y1 = np.amin(projected[1])
    x2 = np.amax(projected[0])
    y2 = np.amax(projected[1])

    img_box = np.array([x1, y1, x2, y2])

    if truncate:
        if not image_shape:
            raise ValueError('Image shape must be provided')

        image_w = image_shape[1]
        image_h = image_shape[0]

        # Discard invalid boxes (outside image space)
        if img_box[0] > image_w or \
                img_box[1] > image_h or \
                img_box[2] < 0 or \
                img_box[3] < 0:
            return None

        # Discard boxes that are larger than 80% of the image width OR height
        if discard and discard_before_truncation:
            img_box_w = img_box[2] - img_box[0]
            img_box_h = img_box[3] - img_box[1]
            if img_box_w > (image_w * 0.8) or img_box_h > (image_h * 0.8):
                return None

        # Truncate remaining boxes into image space
        if img_box[0] < 0:
            img_box[0] = 0
        if img_box[1] < 0:
            img_box[1] = 0
        if img_box[2] > image_w:
            img_box[2] = image_w
        if img_box[3] > image_h:
            img_box[3] = image_h

        # Discard boxes that are covering the the whole image after truncation
        if discard and not discard_before_truncation:
            img_box_w = img_box[2] - img_box[0]
            img_box_h = img_box[3] - img_box[1]
            if img_box_w > (image_w * 0.8) and img_box_h > (image_h * 0.8):
                return None

    # Convert to tf order
    img_box_tf_order = np.array([img_box[1], img_box[0], img_box[3], img_box[2]])

    return img_box_tf_order
