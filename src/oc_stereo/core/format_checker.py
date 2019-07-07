"""
This module checks for the correct data format and dimensions.

The three different anchor formats as well as corner representations are
used throughout the network, this is just a sanity check before format
conversions.

The ObjectLabel format has the following properties:
- ObjectLabel format is well-documented in the object_utils in wavedata
- It is used more for encoding a generic label class.
- It is a format used in evaluation for Kitti.
- This format is useful for interfacing with obj_utils operations. Note: most
  of the obj_utils operation is not optimized for batch operations and it might
  be slow.

The anchor format is the following [x, y, z, dim_x, dim_y, dim_z] (N x 6):
- [x, y, z] are real number along their respective axis in [metres]
- [dim_x, dim_y, dim_z] are real numbers representing the size of
  the box along their respective axis in [metres]
- This form does not encode rotation, and thus is a natural form to use for
  anchor generation and evaluation.

The box_3d format is the following format [x, y, z, l, w, h, ry] (N x 7):
- [x, y, z] are real number along their respective axis in [metres]
- [l, w, h] are real numbers representing the size of the box
- [ry] is the yaw rotation along the y axis of the camera coordinate.
  It is a value between [-pi ... pi].
- This format is used to simply encode a common 3D box with a single rotation
  in the y axis, which makes it useful for BEV operations.

The box_8c format is the following [[x1,...,x8],[y1...,y8], [z1,...,z8]]
(N x 3 x 8):
- [x1, ..., x8] are the corners in the x-axis
- [y1, ..., y8] are the corners in the y-axis
- [z1, ..., z8] are the corners in the z-axis

The box_8co format is the same as box_8c, except that the corners are ordered,
i.e. the order of corners are preserved throughout the conversion.

The box_4c format is the following
[[x1, x2, x3, x4, z1, z2, z3, z4, h1, h2]] (N x 10):
- [x1, x2, x3, x4, z1, z2, z3, z4] are the corners in the xz plane,
    numbered clockwise starting at the top right
- [h1] is the height above the ground plane to the bottom of the box
- [h2] is the height above the ground plane to the top of the box
"""

import numpy as np

from oc_stereo.dataloader.kitti import obj_utils


