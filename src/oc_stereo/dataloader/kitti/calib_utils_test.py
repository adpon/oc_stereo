import os
import unittest

import numpy as np
import torch
from torch.autograd import Variable

from oc_stereo.dataloader.kitti import calib_utils, obj_utils


class Test(unittest.TestCase):

    def test_pc_from_disparity(self):

        # Load in sample information
        calib_dir = os.path.expanduser('~/Kitti/object/training/calib')
        disp_dir = os.path.expanduser('~/Kitti/object/training/disparity_multiscale')
        sample_name = '000050'

        # Read in disparity map
        disp_map = obj_utils.get_disp_map(sample_name, disp_dir)

        # Read in calib
        frame_calib = calib_utils.get_frame_calib(calib_dir, sample_name)
        stereo_calib = calib_utils.get_stereo_calibration(frame_calib.p2,
                                                          frame_calib.p3)

        pc = calib_utils.pc_from_disparity(disp_map, stereo_calib)
        pc = np.expand_dims(np.stack(pc), 0)

        # Create PyTorch Variables
        torch_disp_map = Variable(torch.tensor(np.float32(disp_map)),
                                  requires_grad=False)
        stereo_calib_f = Variable(torch.tensor(np.float32(stereo_calib.f)),
                                  requires_grad=False)
        stereo_calib_b = Variable(torch.tensor(np.float32(stereo_calib.baseline)),
                                  requires_grad=False)
        stereo_calib_center_u = Variable(torch.tensor(np.float32(stereo_calib.center_u)),
                                         requires_grad=False)
        stereo_calib_center_v = Variable(torch.tensor(np.float32(stereo_calib.center_v)),
                                         requires_grad=False)

        # Move to CUDA
        torch_disp_map, stereo_calib_f, stereo_calib_b, stereo_calib_center_u, \
        stereo_calib_center_v = \
            torch_disp_map.cuda(), stereo_calib_f.cuda(), stereo_calib_b.cuda(), \
            stereo_calib_center_u.cuda(), stereo_calib_center_v.cuda()

        # Add batch dims
        torch_disp_map = torch.unsqueeze(torch_disp_map, 0)

        pc_torch = calib_utils.torch_pc_from_disparity(torch_disp_map, stereo_calib_f,
                                                       stereo_calib_b,
                                                       stereo_calib_center_u, stereo_calib_center_v)

        pc_torch_to_np = pc_torch.cpu().numpy()

        np.testing.assert_array_almost_equal(pc, pc_torch_to_np, decimal=5)


if __name__ == 'main':
    unittest.main()
