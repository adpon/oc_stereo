import os
import sys

import numpy as np

from oc_stereo.builders.dataset_builder import DatasetBuilder
from oc_stereo.dataloader.kitti import calib_utils, depth_map_utils, obj_utils


def main():
    """Save object detection disparity maps using depth dataset"""

    ####################
    # Options
    ####################

    dataset = DatasetBuilder.build_kitti_dataset(
        # DatasetBuilder.KITTI_TRAIN
        # DatasetBuilder.KITTI_VAL
        DatasetBuilder.KITTI_TRAINVAL
    )

    depth_version = 'multiscale'
    data_split_dir = 'training'  # test
    output_dir = os.path.expanduser('~/Kitti/object/{}/disparity_{}'.format(data_split_dir,
                                                                            depth_version))
    ####################

    if os.path.isdir(output_dir):
        pass
    else:
        os.makedirs(output_dir)

    num_samples = dataset.num_samples
    calib_dir = dataset.calib_dir

    dataset.depth_version = depth_version
    depth_dir = os.path.expanduser('~/Kitti/object/training/depth_2_{}'.format(depth_version))

    for sample_idx in range(num_samples):

        sys.stdout.write('\r{} / {}'.format(sample_idx + 1, num_samples))

        sample_name = dataset.sample_list[sample_idx].name

        # Load in depth map
        try:
            depth_map = obj_utils.get_depth_map(sample_name, depth_dir)
        except TypeError:
            continue
        # depth_mask = depth_map != 0

        # Get calib
        frame_calib = calib_utils.get_frame_calib(calib_dir, sample_name)
        stereo_calib = calib_utils.get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # Calculate disparity
        baseline = stereo_calib.baseline
        focal_length = frame_calib.p2[0, 0]

        # Disparity
        disp = (baseline * focal_length) / depth_map
        disp[disp == np.inf] = 0

        # Save disparity
        output_path = output_dir + '/' + sample_name + '.png'
        depth_map_utils.save_depth_map(output_path, disp)


if __name__ == '__main__':
    main()