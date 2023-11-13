
from datasets.dataset import ImageDataset
from datasets.transforms import *
from optimizers import LocalBA, PoseGraphOptimization
from slamStructure import SLAMStructure
from tracking import Tracking
from modules.depth_estimation import *
from modules.pose_guessing import *
from modules.point_resampling import *
from modules.loop_detectors import *
from cotracker.predictor import CoTrackerPredictor
from modules.loop_closure_test import *
from modules.keyframe_select import *
from modules.mapping import *

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
import argparse
import random
import time
import threading
import logging

from scripts.logging_config import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

class SLAM:
    def __init__(self, dataset, args) -> None:
        self.args = args
        self.dataset = dataset
        self.slam_structure = SLAMStructure(dataset, name=args.name, output_folder=args.output_folder)
        # Create LocalBA object for bundle adjustment
        self.localBA = LocalBA(self.slam_structure, BA_sparse_solver=not args.dense_ba, BA_opt_iters=args.tracking_ba_iterations, BA_verbose=args.verbose_ba)
        self.PGO = PoseGraphOptimization(self.slam_structure)
        # Load tracking module and associated components
        # Pose guesser component
        self.pose_guesser =  PoseGuesserLastPose()
        # Depth estimation component
        self.depth_estimator = DepthEstimatorConstant(args.depth_scale)
        # Point sampling component
        self.point_sampler = PointResamplerR2D2(args.tracked_point_num_min, args.tracked_point_num_max, args.minimum_new_points)    
        # Point tracking component
        self.cotracker = CoTrackerPredictor(checkpoint="./trained_models/cotracker/"+args.cotracker_model+".pth").to(device=args.target_device)
        self.cotracker.eval()
        # Create tracking module
        self.tracking_module = Tracking(self.depth_estimator, self.point_sampler, self.cotracker, args.target_device, cotracker_window_size = args.cotracker_window_size, BA=self.localBA)
        # Create loop closure module
        self.loop_close_module = LoopClosureR2D2(self.cotracker, self.slam_structure, self.dataset, args.start_idx, args.end_idx)
        # Keyframe selection module
        # self.keyframe_module = KeyframeSelectSubsample(self.dataset, self.depth_estimator, self.slam_structure, self.localBA, args.keyframe_subsample)
        self.keyframe_module = KeyframeSelectFeature(self.dataset, self.depth_estimator, self.slam_structure, self.localBA)
        # Mapping module
        self.mapping_module = mapping(self.slam_structure, self.localBA, args.point_resample_cooldown)
        # Current section and keyframes 
        self.current_section = []
        # store loops
        self.loops = []
        self.num_loops = 0
        # For calculating tracking FPS (how long does frame localization take on average)
        self.total_tracking_time = 0
        self.tracking_counter = 0
        # For calculating mapping FPS (how long does mapping take on average)
        self.total_mapping_time = 0
        self.mapping_counter = 0
        self.completed_sections = 0
        # For calculating effective update FPS (how many frame updates does the pipeline provide per second)
        self.update_counter = 0
        self.total_update_time = 0
        self.update_start_time = 0
        
        self.need_sample = False
        
    def initialize(self):
        for frame_idx in frames_to_process:
            last_poses = self.slam_structure.get_previous_poses(10)
            if len(slam.slam_structure.poses.keys()) == 0:
                intrinsics = dataset[frame_idx]['intrinsics'].detach().cpu().numpy()
                
                pose = slam.pose_guesser(last_poses)

                # Add first frame and make into keyframe
                slam.slam_structure.add_frame(frame_idx, pose, intrinsics)
                slam.keyframe_module.run(frame_idx)
            else:
                # Retrieve data to add new frame
                intrinsics = dataset[frame_idx]['intrinsics'].detach().cpu().numpy()
                pose = slam.pose_guesser(last_poses)

                # Add new frame
                slam.slam_structure.add_frame(frame_idx, pose, intrinsics)

            
            # Add frame to current section
            self.current_section.append(frame_idx)
            # Check if frame buffer is full.
            if len(self.current_section) < self.args.section_length:
                continue
            # Start mapping
            self.mapping()

            if self.completed_sections == 2:
                # This was the map initialization, start update counter
                self.update_start_time = time.time()
                print("Map initizalized.")
                break
    
    def localization(self):
        # If localization is enabled and map is initiallized, localize frame
        if self.completed_sections > 1:
            tracking_start_time = time.time()
            localized_frame = slam.slam_structure.poses[frame_idx][0]
            # Skip localization if disabled, good for test runs
            if not args.no_localize:
                # Run a minimal point tracking to obtain localization correspondences
                section_to_track = list(self.slam_structure.poses.keys())[-self.args.past_frame_size:]
                # start_frame = max(0, self.args.past_frame_size - len(self.current_section) - 1)
                start_frame = 0
                if self.args.verbose:
                    print("Tracking frame: ", frame_idx)
                    print("Current section: ", self.current_section)
                    print("Section to track: ", section_to_track)
                    print("Tracking start index: ", section_to_track[start_frame])

                self.tracking_module.process_section(section_to_track, self.dataset, self.slam_structure, 
                                            sample_new_points=False, 
                                            start_frame=start_frame,
                                            maximum_track_num=self.args.localization_track_num)
                
                last_pose_points = self.slam_structure.pose_point_map[self.current_section[-1]]
                current_pose_points = self.slam_structure.pose_point_map[frame_idx]
                # logger.info(f"Tracking frame: {frame_idx}")
                # logger.info(f"Section to track: {section_to_track}")
                # logger.info(f"Tracking start index: {section_to_track[start_frame]}")
                logger.info(f'last frame:{self.current_section[-1]}: {len(last_pose_points[:self.args.localization_track_num])},'\
                            f'current frame: {len(current_pose_points)}, ratio: {len(current_pose_points)/len(last_pose_points[:self.args.localization_track_num])}')
                if len(current_pose_points) < 0.5 * self.args.localization_track_num:
                    # This allows to sample more points when the points lost are too many
                    self.need_sample = True
                # Localize frame
                localized_frame = self.slam_structure.localize_frame(frame_idx, 
                                                                update_pose=self.args.update_localized_pose,
                                                                ransac=self.args.ransac_localization)

            # Tracking done and new update available
            current_time = time.time()

            self.update_counter += 1
            self.total_update_time += current_time - self.update_start_time

            self.tracking_counter += 1
            self.total_tracking_time += current_time - tracking_start_time

            # Print update
            if self.args.verbose:
                print("Current frame: ", localized_frame[:3, 3])
                print("Running section: ", self.current_section)
                print("Running FPS: ", self.update_counter/self.total_update_time)
                if self.tracking_counter > 0:  print("Running Tracking FPS: ", self.tracking_counter/self.total_tracking_time)
                if self.mapping_counter > 0: print("Running Mapping FPS: ", self.mapping_counter/self.total_mapping_time)
                
            self.update_start_time = time.time()
            
    def mapping(self):
        print("Mapping ...")
        mapping_start_time = time.time()
        # Section full, start MAPing
        # Remove all existing correspondences (likely to be faulty), except for frist frame
        self.mapping_module.remove_correspondences(self.current_section)

        # Obtain new consistent set of point correspondences
        # Track the points throughout the section, keep only covisible points
        # section_to_track = np.copy(self.current_section)
        section_to_track = self.current_section.copy()
        # update correspondences for each frame to slam_structure
        self.tracking_module.process_section(section_to_track, self.dataset, self.slam_structure, sample_new_points=True,
                                        start_frame=0,
                                        maximum_track_num=args.tracked_point_num_max)
        # Decide to make frames into new keyframes
        self.keyframe_module.new_keyframe_counter = 0
        for idx in self.current_section[1:]:
            # Keyframe decision, adding BA pose-point edges
            self.keyframe_module.run(idx)
            # self.loop_closure(idx)
            # TODO for after paper: Add loop closure here
        
        # If there are new keyframes, run local BA
        # print('new_keyframe_counter:', self.keyframe_module.new_keyframe_counter)
        if self.keyframe_module.new_keyframe_counter > 0:
            #TODO:local BA logical adjust 
            self.mapping_module.local_BA(args.local_ba_size, args.tracking_ba_iterations, self.keyframe_module.new_keyframe_counter)
            bad_measurements = self.localBA.get_bad_measurements()
            self.slam_structure.remove_measurement(bad_measurements)
            #     # # Loop closing
            # if len(self.loops) > self.num_loops:
            #     self.num_loops = len(self.loops)
            #     print('loops:', self.loops) 
            #     print('Start loop closing .........................................')
            #     logger.info(f'Loops for section starts from {frame_idx}: {self.loops}')
            #     # PGO
            #     loop_mes_list = slam.loop_close_module.find_motion(slam.loops)
            #     slam.PGO.set_data(loop_mes_list)
            #     slam.PGO.run_pgo()
            #     # update BA with poses and points after PGO
            #     slam.localBA.update_ba_data()
            
        # Mapping done
        current_time = time.time()
        self.mapping_counter += 1
        self.total_mapping_time += current_time - mapping_start_time
        
        # Update section
        self.current_section = self.current_section[-1:]
        self.completed_sections += 1
        
    def loop_detect(self, frame_idx):
        # detect loops for each frame
        loop = self.loop_close_module.find_loops(frame_idx) 
        if loop is not None and loop[0] in self.slam_structure.keyframes:
            self.keyframe_module.setKeyframe(frame_idx)
            self.loops.append(loop)
    
if __name__ == "__main__":

    # TODO: Store arguments and code after execution

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
    parser.add_argument("--lumen_mask_low_threshold", default=0., type=float, help="mask out pixels that are too dark")
    parser.add_argument("--lumen_mask_high_threshold", default=1., type=float, help="mask out pixels that are too bright")

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
    parser.add_argument('--depth_scale',  default=1.0, type=float, help='scaling factor for depth esimates')
    parser.add_argument("--depth_network_path", default="", type=str, help="path to depth network, only required for depth estimation using the network")
    parser.add_argument('--point_sampler', type=str, choices=['uniform', 'sift', 'orb', 'r2d2', 'density','akaze'], default='uniform', 
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
    parser.add_argument('--cotracker_model', type=str, choices=['cotracker_stride_4_wind_8', 'cotracker_stride_4_wind_12', 'cotracker_stride_8_wind_16'], default='cotracker_stride_4_wind_8', 
                        help='cotracker model to be used')
    parser.add_argument('--cotracker_window_size', type=int, choices=[8, 12, 16], default=8, 
                        help='window size of the current co-tracker model. Choos appropriately.')

    # Stuff for BA
    parser.add_argument("--dense_ba", action='store_true', help="use dense instead of sparse bundle adjustment")
    parser.add_argument("--verbose_ba", action='store_true', help="output additional information for bundle adjustment")
    parser.add_argument("--tracking_ba_iterations", default=20, type=int, help="number of ba iterations after tracking")
    parser.add_argument("--local_ba_size", default=10, type=int, help="maximum number of keyframes to include in local BA, -1 to use all keyframes")


    # Parse arguments
    args = parser.parse_args()

    # Argument consistency checks
    # TODO: More potential consistency checks

    # If processing subset, start_idx, end_idx and image_subsample must be set.
    if args.process_subset:
        assert args.start_idx != -1
        assert args.end_idx != -1
        assert args.image_subsample != -1

    # If using pretrained depth estimator, path needs to be set
    if args.depth_estimator == 'depth_network':
        assert args.depth_network_path != ''

    assert args.keyframe_subsample < args.section_length

    # Set seed for reproduceability
    random.seed(args.seed)     # python random generator
    np.random.seed(args.seed)  # numpy random generator

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    torch.autograd.set_detect_anomaly(mode=False)
    torch.autograd.profiler.profile(enabled=False)

    #################################################
    # Dataset/Data management and bundle adjustment #
    #################################################

    # Load dataset

    """
    # TODO: Mask out blurry image areas?
    dataset_ref = ImageDataset(args.data_root, transform = transforms.Compose([
                                                            SampleToTensor(),
                                                            RescaleImages((args.img_height, args.img_width))
                                                        ]))
                            
    dataset_ref_sample = dataset_ref[list(dataset_ref.images.keys())[0]]
    """

    composed_transforms = transforms.Compose([SampleToTensor(),
                                            RescaleImages((args.img_height, args.img_width)),
                                            MaskOutLuminosity(threshold_high=args.lumen_mask_high_threshold, threshold_low=args.lumen_mask_low_threshold),
                                            SampleToDevice(args.target_device)])
    dataset = ImageDataset(args.data_root, transform=composed_transforms)

    # Determine frames to process
    frames_to_process = list(dataset.images.keys())
    if args.process_subset:
        if args.start_idx > args.end_idx:
            frames_to_process = list(filter(lambda frame_idx: frame_idx in range(args.end_idx, args.start_idx, args.image_subsample), frames_to_process))
            frames_to_process = frames_to_process[::-1]
        else:
            frames_to_process = list(filter(lambda frame_idx: frame_idx in range(args.start_idx, args.end_idx, args.image_subsample), frames_to_process))



        
    if args.start_idx and args.end_idx:
        print("Number of frames to process: ", args.end_idx - args.start_idx + 1)
    else:
        print("Number of frames to process: ", len(dataset))

###################################################################################################################
#                                                   SLAM MAIN LOOP                                                #
###################################################################################################################
slam = SLAM(dataset, args)
# Main SLAM loop
slam.initialize()

for frame_idx in tqdm(frames_to_process):
    
    if frame_idx in slam.slam_structure.poses.keys():
        continue
    # Add frame to slam structure
    last_poses = slam.slam_structure.get_previous_poses(10)

    # Retrieve data to add new frame
    intrinsics = dataset[frame_idx]['intrinsics'].detach().cpu().numpy()
    pose = slam.pose_guesser(last_poses)

    # Add new frame
    slam.slam_structure.add_frame(frame_idx, pose, intrinsics)
    
        
    ###########################################################################################################
    #                                                   localization                                          #
    ###########################################################################################################
    # loop_thread = threading.Thread(target=slam.loop_closure, args=(frame_idx,))
    # loop_thread.start()
    # slam.loop_detect(frame_idx)
    # If localization is enabled and map is initiallized, localize frame
    slam.localization()
    
    # Add frame to current section
    slam.current_section.append(frame_idx)
    # Check if frame buffer is full.
    if len(slam.current_section) < slam.args.section_length and not slam.need_sample:
        # logger.info(f'current_section: {slam.current_section}')
        continue
    slam.need_sample = False
    ###########################################################################################################
    #                                                   Mapping                                               #
    ###########################################################################################################
    slam.mapping()
    # # Loop closing
    # if len(slam.loops) > slam.num_loops:
    #     slam.num_loops = len(slam.loops)
    #     print('loops:', slam.loops) 
    #     print('Start loop closing .........................................')
    #     logger.info(f'Loops for section starts from {frame_idx}: {slam.loops}')
        
    #     # PGO
    #     loop_mes_list = slam.loop_close_module.find_motion(slam.loops)
    #     slam.PGO.set_data(loop_mes_list)
    #     slam.PGO.run_pgo()
    #     # update BA with poses and points after PGO
    #     slam.localBA.update_ba_data()
        
    # loop_thread.join()
# Filter outliers in reconstruction
# NOTE: ONLY FILTER AFTER EVERYTHING IS OPTIMIZED, DOES NOT UPDATE 
slam.slam_structure.filter(min_view_num=2, reprojection_error_threshold=10)

slam.slam_structure.save_visualizations()
# Save data and visualizations
slam.slam_structure.save_data(dataset, 
                         update_fps=slam.update_counter/slam.total_update_time,
                         tracking_fps=slam.tracking_counter/slam.total_tracking_time,
                         mapping_fps=slam.mapping_counter/slam.total_mapping_time)

print("Done.")