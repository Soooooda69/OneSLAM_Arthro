import argparse

class SlamOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Python file for running the SLAM pipeline',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Data loading
        parser.add_argument("--data_root", required=True, type=str, help="path to dataset")
        parser.add_argument("--process_subset", action='store_true', help="use start_idx/end_idx/image_subsample to define a subset of processed frames in dataset")
        parser.add_argument("--start_idx", default=-1, type=int, help="starting frame index for processing")
        parser.add_argument("--end_idx", default=-1, type=int, help="end frame index for processing")
        parser.add_argument("--image_subsample", default=-1, type=int, help="custom image subsample factor")
        parser.add_argument("--img_width", default=-1, type=int, help="width to rescale images to, -1 for no scaling")
        parser.add_argument("--img_height", default=-1, type=int, help="height to rescale images to, -1 for no scaling")
        parser.add_argument("--lumen_mask_low_threshold", default=0.05, type=float, help="mask out pixels that are too dark")
        parser.add_argument("--lumen_mask_high_threshold", default=0.95, type=float, help="mask out pixels that are too bright")

        # General Stuff
        parser.add_argument("--target_device", default='cuda:0', type=str, help="GPU to run process on")
        parser.add_argument("--name", default='', type=str, help="name of current experiment, leave empty to use current time")
        parser.add_argument('--output_folder',  default='./experiments', type=str, help="where to store output of slam run")
        parser.add_argument("--seed", default=42, type=int, help="seed for reproducability")
        parser.add_argument("--verbose", action='store_true', help="print what is happening")
        parser.add_argument("--no_localize", action='store_true', help="do not run localization (does not affect mapping procedure)")


        # Stuff for tracking
        parser.add_argument("--section_length", default=13, type=int, help="how many frames to buffer before running section point tracking")
        parser.add_argument("--past_frame_size", default=5, type=int, help="how many past frames to include in point tracking")
        parser.add_argument('--keyframe_decision', type=str, choices=['subsample', 'orb'], default='subsample', 
                            help='How to make keyframe decision'
                                'subsample: subsample every keyframe_subsample pose, default 4'
                                'orb: use keyframe decision process from orb')
        parser.add_argument("--keyframe_subsample", default=4, type=int, help="how often to sample a keyframe")
        parser.add_argument('--pose_guesser', type=str, choices=['last_pose', 'constant_velocity', 'gt_pose'], default='last_pose', 
                            help='how to obtain initial pose guess'
                                'last_pose: use last pose as initial guess'
                                'constant_velocity: use constant velocity model to obtain initial pose guess')
        parser.add_argument("--gt_pose_path", default='', type=str, help="name of ground truth pose path, only required for gt_pose pose guesser")
        parser.add_argument('--depth_estimator', type=str, choices=['constant', 'luminosity', 'depth_network'], default='constant', 
                            help='how to obtain depth estimates for point projection'
                                'constant: use constant depths'
                                'luminosity: use brightness as initial depth guess'
                                'depth_network: use pretrained depth estimation network')
        parser.add_argument('--depth_scale',  default=1, type=float, help='scaling factor for depth esimates')
        parser.add_argument("--depth_network_model", default="", type=str, help="path to depth network, only required for depth estimation using the network")
        parser.add_argument('--point_sampler', type=str, choices=['uniform', 'sift', 'orb', 'r2d2', 'density','akaze'], default='r2d2', 
                            help='how to sample new points at each section start'
                                'uniform: uniform point sampling'
                                'sift: sample sift keypoints'
                                'orb: sample orb keypoints'
                                'r2d2: sample r2d2 keypoints'
                                'density: sample new points based on existing point density')
        parser.add_argument('--tracked_point_num_min',  default=200, type=int, help='Minimum number of point to track per section.')
        parser.add_argument('--tracked_point_num_max',  default=2000, type=int, help='Maximum number of point to track per section.')
        parser.add_argument('--localization_track_num',  default=50, type=int, help='Maximum number of points on localization.')
        parser.add_argument("--update_localized_pose", action='store_true', help="update a localized pose in the SLAM datastructure")
        parser.add_argument("--ransac_localization", action='store_true', help="use ransac pnp to localize pose")
        parser.add_argument('--minimum_new_points',  default=0, type=int, help='Minimum number of new points to sample after every section.')
        parser.add_argument('--point_resample_cooldown',  default=1, type=int, help='How many sections to wait before resampling points again')
        parser.add_argument('--cotracker_model', type=str, choices=['cotracker2','cotracker_stride_4_wind_8', 'cotracker_stride_4_wind_12', 'cotracker_stride_8_wind_16'], default='cotracker_stride_4_wind_8', 
                            help='cotracker model to be used')
        parser.add_argument('--cotracker_window_size', type=int, choices=[8, 12, 16], default=8, 
                            help='window size of the current co-tracker model. Choos appropriately.')

        # Stuff for BA
        parser.add_argument("--dense_ba", action='store_true', help="use dense instead of sparse bundle adjustment")
        parser.add_argument("--verbose_ba", action='store_true', help="output additional information for bundle adjustment")
        parser.add_argument("--tracking_ba_iterations", default=20, type=int, help="number of ba iterations after tracking")
        parser.add_argument("--local_ba_size", default=10, type=int, help="maximum number of keyframes to include in local BA, -1 to use all keyframes")

        # Parse arguments
        self.args = parser.parse_args()
        
    