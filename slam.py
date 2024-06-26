
from datasets.dataset import ImageDataset
from datasets.transforms import *
from optimizers import LocalBA, PoseGraphOptimization
from slamStructure import SLAMStructure
from tracking import Tracking
from modules.depth_estimation import *
from modules.pose_guessing import *
from modules.point_resampling import *
from modules.loop_detectors import *
from cotracker.predictor import * #CoTrackerPredictor, CoTrackerOnlinePredictor
from modules.loop_closure_test import *
from modules.keyframe_select import *
from modules.mapping import *
from misc.covisibility import CovisibilityGraph
from misc.components import Measurement
from slamOptions import SlamOptions

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
logger = logging.getLogger('logfile.txt')

class SLAM:
    def __init__(self, dataset, args) -> None:
        self.args = args
        self.dataset = dataset
        # Create CovisibilityGraph
        self.graph = CovisibilityGraph()
        # Create SLAM data Structure
        self.slam_structure = SLAMStructure(dataset, self.graph, name=args.name, output_folder=args.output_folder)
        # Create LocalBA object for bundle adjustment
        self.localBA = LocalBA(self.slam_structure, BA_sparse_solver=not args.dense_ba, BA_opt_iters=args.tracking_ba_iterations, BA_verbose=args.verbose_ba)
        self.PGO = PoseGraphOptimization(self.slam_structure)
                
        # Load tracking module and associated components
        # Pose motion model
        self.motion_model = MotionModel()
        # Pose guesser component
        self.pose_guesser =  PoseGuesserLastPose()
        # Depth estimation component
        self.depth_estimator = DepthEstimatorConstant(args.depth_scale)
        # self.depth_anything = DepthEstimatorDepthAnything(args.depth_network_model)
        # Point sampling component
        self.point_sampler = PointResamplerR2D2(args.tracked_point_num_min, args.tracked_point_num_max, args.minimum_new_points)    
        # Point tracking component
        self.cotracker = CoTrackerPredictor(checkpoint="./trained_models/cotracker/"+args.cotracker_model+".pth").to(device=args.target_device)
        self.cotracker.eval()
        # Create tracking module
        self.tracking_module = Tracking(self.depth_estimator, self.point_sampler, self.cotracker, args.target_device, args.cotracker_window_size, self.localBA, self.dataset)
        # Create loop closure module
        self.loop_close_module = LoopClosureR2D2(self.cotracker, self.slam_structure, self.dataset, args.start_idx, args.end_idx)
        # Keyframe selection module
        # self.keyframe_module = KeyframeSelectSubsample(self.dataset, self.depth_estimator, self.slam_structure, self.localBA, args.keyframe_subsample)
        self.keyframe_module = KeyframeSelectFeature(self.dataset, self.depth_estimator, self.slam_structure, self.localBA)
        # Mapping module
        self.mapping_module = mapping(self.slam_structure, self.localBA, self.graph, args.local_ba_size)
        # Current section and keyframes 
        self.current_section = []
        # store loops
        # self.loops = []
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
        
        self.is_initialized = False
        self.need_sample = False
        self.sample_idx = -1
        # self.tracking_is_ok = []
        self.tracking_is_ok = True
        
    def initialize(self):
        print('Initializing map ...')
        # for frame_idx in frames_to_process:
        if not self.is_initialized:     
            # Check if frame buffer is full.
            if len(self.current_section) < self.args.section_length:
                return    
            
            # Start mapping
            section_to_track = self.current_section.copy()
            # update correspondences for each frame to slam_structure
            self.tracking_module.process_section(section_to_track, self.dataset, self.slam_structure, sample_new_points=True,
                                            start_frame=0,
                                            maximum_track_num=args.tracked_point_num_max,
                                            mapping=True,
                                            initialize=True)
            
            self.sample_idx = self.current_section[-1]
            # Force 1st frame to be keyframe
            if not self.slam_structure.current:
                frame_0 = self.slam_structure.all_frames[frames_to_process[0]]
                # Add to keyframes
                self.keyframe_module.setKeyframe(frames_to_process[0])
                keyframe_0 = self.slam_structure.key_frames[frames_to_process[0]]
                keyframe_0.set_fixed(True)
                
                self.slam_structure.current = keyframe_0
                self.slam_structure.reference = keyframe_0
                self.slam_structure.preceding = keyframe_0
                
            # Decide keyframes    
            for idx in self.current_section:
                # Keyframe decision, adding BA pose-point edges
                self.keyframe_module.run(idx)
                        
            # The number of keyframes to initialize the map is IMPORTANT!
            init_num = 2
            if self.keyframe_module.new_keyframe_counter >= init_num and len(self.slam_structure.key_frames)!=1:
                logger.info(f'Initialize keyframes:{self.keyframe_module.new_keyframes[:2]}')
                self.mapping_module.full_BA()
                
                # Update covisibility graph & motion model
                for keyframe_idx in self.keyframe_module.new_keyframes:
                    keyframe = self.slam_structure.key_frames[keyframe_idx]
                    self.graph.add_keyframe(keyframe)
                    keyframe.update_reference(self.slam_structure.reference)
                    keyframe.update_preceding(self.slam_structure.preceding)
                    
                    frame = self.slam_structure.all_frames[keyframe_idx]
                    frame.update_pose(keyframe.pose.matrix())
                    self.slam_structure.create_points(keyframe)
                    self.motion_model.update_pose(frame.timestamp, frame.pose.position(), frame.pose.orientation())
                    
                # Reset new keyframe counter
                self.keyframe_module.resetNewKeyframeCounter()
                
                # Log info
                logger.info(f'number of keyframes to initialize:{init_num}')
                logger.info(f'map points:{len(self.slam_structure.map_points)}')
                logger.info(f'total edges: {len(self.localBA.BA.active_edges())}')

                self.update_start_time = time.time()
                print("Map initizalized.")
                self.current_section = self.current_section[-1:]
                self.completed_sections += 1
                slam.is_initialized = True
            
            # Update section
            self.current_section = self.current_section[-1:]
            self.completed_sections += 1
            
    def localization(self):
        # If localization is enabled and map is initiallized, localize frame
        # if self.completed_sections > 0:
            tracking_start_time = time.time()
            # localized_frame = slam.slam_structure.poses[frame_idx][0]
            # Skip localization if disabled, good for test runs
            if not args.no_localize:
                # Run a minimal point tracking to obtain localization correspondences
                section_to_track = list(self.slam_structure.all_frames.keys())[-self.args.past_frame_size:]
                # start_frame = max(0, self.args.past_frame_size - len(self.current_section) - 1)
                # start_frame = 0
                # if self.args.verbose:
                    # print("Tracking frame: ", frame_idx)
                    # print("Current section: ", self.current_section)
                print("Section to track: ", section_to_track)
                    # print("Tracking start index: ", section_to_track[start_frame])

                self.tracking_module.process_section(section_to_track, self.dataset, self.slam_structure, 
                                            sample_new_points=False, 
                                            start_frame=0,
                                            maximum_track_num=self.args.localization_track_num)
                
                # current_frame from slam_structure.all_frames
                current_frame = self.slam_structure.lc_frames[frame_idx]
                prev_frame = self.slam_structure.lc_frames[frame_idx - 1]
                first_frame = self.slam_structure.all_frames[section_to_track[0]]
                # Calculate motion of tracked points
                intersection_motion = []
                for point_id in current_frame.feature.keypoints_ids:
                    last_point = first_frame.feature.keypoints_info[point_id][0]
                    current_point = current_frame.feature.keypoints_info[point_id][0]
                    motion = np.linalg.norm(last_point - current_point)
                    intersection_motion.append(motion)
                # This allows to sample more points(start mapping) when points motion are fast
                if np.mean(intersection_motion) > 20 or \
                    len(current_frame.feature.keypoints) < self.args.localization_track_num:
                    self.need_sample = True
                
                # # # update current frame
                # # self.slam_structure.current = current_frame
                # if len(current_frame.feature.keypoints) < 0.4 * self.args.localization_track_num:
                #     # This allows to sample more points(start mapping) when the points lost are too many
                #     self.need_sample = True
                    
                # Localize frame with PnP
                localized_pose = self.slam_structure.localize_frame(current_frame, 
                                                                update_pose=True,
                                                                ransac=self.args.ransac_localization)
                localized_pose = g2o.Isometry3d(localized_pose)
                # self.motion_model.update_pose(
                # current_frame.timestamp, localized_pose.position(), localized_pose.rotation())
                # print(f"Localized frame {current_frame.idx} with {localized_pose.position()}")
                
                self.slam_structure.lc_frames = {}
            
            # Tracking done and new update available
            current_time = time.time()
            self.update_counter += 1
            self.total_update_time += current_time - self.update_start_time
            self.tracking_counter += 1
            self.total_tracking_time += current_time - tracking_start_time

            # Print update
            if self.args.verbose:
                # print("Current frame: ", localized_frame[:3, 3])
                print("Running section: ", self.current_section)
                # print("Running FPS: ", self.update_counter/self.total_update_time)
                if self.tracking_counter > 0:  print("Running Tracking FPS: ", self.tracking_counter/self.total_tracking_time)
                if self.mapping_counter > 0: print("Running Mapping FPS: ", self.mapping_counter/self.total_mapping_time)
                
            self.update_start_time = time.time()
    
        
    def mapping(self):
        # print("Tracking ...")
        mapping_start_time = time.time()
        section_to_track = self.current_section.copy()
        print('section_to_map:', section_to_track)
        
        # update correspondences for each frame to slam_structure
        self.tracking_module.process_section(section_to_track, self.dataset, self.slam_structure, sample_new_points=self.need_sample,
                                        start_frame=0,
                                        maximum_track_num=args.tracked_point_num_max,
                                        mapping=True)
        
        current_frame = self.slam_structure.all_frames[self.current_section[-1]]

        self.need_sample = False
        # Decide to make frames into new keyframes
        for idx in self.current_section:
            # Keyframe decision, adding BA pose-point edges
            self.keyframe_module.run(idx)
        
        # Add loop frame to keyframes        
        if len(self.loop_close_module.loops) > 0:
            if self.loop_close_module.loops[1] not in self.slam_structure.key_frames.keys():
                self.keyframe_module.setKeyframe(self.loop_close_module.loops[1])
                self.slam_structure.key_frames = dict(sorted(self.slam_structure.key_frames.items(), key=lambda x: x[0]))
            keyframe = self.slam_structure.key_frames[self.loop_close_module.loops[1]]
            keyframe.set_loop(self.slam_structure.key_frames[self.loop_close_module.loops[0]], g2o.Isometry3d(np.identity(4)))
            self.loop_close_module.loops = []
            
        # If there are new keyframes, run local BA
        if self.keyframe_module.new_keyframe_counter > 0:
            self.mapping_module.local_BA(args.tracking_ba_iterations, self.keyframe_module.new_keyframe_counter)
         
            # Reset new keyframe counter
            self.keyframe_module.resetNewKeyframeCounter()
            
        # Mapping done
        current_time = time.time()
        self.mapping_counter += 1
        self.total_mapping_time += current_time - mapping_start_time
             
        # Update section
        self.current_section = self.current_section[-1:]
        self.completed_sections += 1
          
    def loop_detect(self, frame_idx):
        # detect loops for each frame
        self.loop_close_module.loop_coolDown -= 1
        loop = self.loop_close_module.find_loops(frame_idx) 
        if loop is not None:
            # self.loops.append(loop)
            # if all(keyframe_idx in self.slam_structure.key_frames.keys() for keyframe_idx in loop):
            if loop[0] in self.slam_structure.key_frames.keys():
                self.loops.append(loop)
                logger.info(f'Detected loop: {self.loops[-1][1]} and {self.loops[-1][0]}')
    
    def loop_closing(self):
        logger.info('Start loop closing .........................................')
        # PGO
        loop_mes = self.loop_close_module.find_motion(slam.loops[-1])
        # self.loop_close_module.loop_mes_list.append(loop_mes)
        # self.slam_structure.key_frames[loop_mes[1]].set_loop(self.slam_structure.key_frames[loop_mes[0]], loop_mes[2])
        self.PGO.set_data(self.loop_close_module.loop_mes_list)
        self.PGO.run_pgo()
        # update BA with poses and points after PGO
        self.localBA.update_ba_data()
        
        # BA to fix scale
        # for idx in self.slam_structure.keyframes:
        for keyframe in self.slam_structure.key_frames.values():
            if keyframe.idx > loop_mes[0] and keyframe.idx <= loop_mes[1]:
                self.localBA.BA.fix_pose(keyframe.idx, fixed=False)
            else:
                self.localBA.BA.fix_pose(keyframe.idx, fixed=True)
        first_keyframe = next(iter(self.slam_structure.key_frames.values()))            
        self.localBA.BA.fix_pose(first_keyframe.idx, fixed=True)
        self.localBA.run_ba(opt_iters=20)
        # logger.info(f'total: {len(self.localBA.BA.active_edges())}')
        # # self.localBA.get_bad_measurements()

        # After loop closing, wait for a while to avoid the next loop closing
        self.loop_close_module.loop_coolDown = 100
    
if __name__ == "__main__":
    
    # TODO: Store arguments and code after execution
    args = SlamOptions().args
    logger.info(f'experiment name: {args.name}, start-end: {args.start_idx}-{args.end_idx}, section_length: {args.section_length}, past_frame_size: {args.past_frame_size}, local_ba_size: {args.local_ba_size}, tracked_point_num_min: {args.tracked_point_num_min}')

    # If processing subset, start_idx, end_idx and image_subsample must be set.
    if args.process_subset:
        assert args.start_idx != -1
        assert args.end_idx != -1
        assert args.image_subsample != -1

    # If using pretrained depth estimator, path needs to be set
    if args.depth_estimator == 'depth_network':
        assert args.depth_network_model != ''

    # assert args.keyframe_subsample < args.section_length

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
                                            # MaskOutLuminosity(threshold_high=args.lumen_mask_high_threshold, threshold_low=args.lumen_mask_low_threshold),
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

# slam.depth_anything_test()
# slam.unitest()
# breakpoint()

# Main SLAM loop


for frame_idx in tqdm(frames_to_process):
    # if frame_idx in slam.slam_structure.all_frames.keys():
    #     continue
    # Add frame to slam structure
    last_pose = slam.slam_structure.get_previous_poses(10)
    # pose = slam.pose_guesser(last_pose)
    pose = np.identity(4)
    # Add new frame
    current_frame = slam.slam_structure.add_frame(frame_idx, pose)
    slam.motion_model.update_pose(current_frame.timestamp, last_pose.position(), last_pose.rotation())
    predicted_pose, _ = slam.motion_model.predict_pose(current_frame.timestamp)
    current_frame.update_pose(predicted_pose)
    # Add frame to current section
    slam.current_section.append(frame_idx)
    # Run initialization
    if not slam.is_initialized:
        slam.initialize()
        continue
    # slam.slam_structure.save_visualizations()
    # breakpoint()
    ###########################################################################################################
    #                                                   localization                                          #
    ###########################################################################################################
    
    # slam.loop_detect(frame_idx)
    # If localization is enabled and map is initiallized, localize frame
    slam.loop_close_module.loop_detect(frame_idx)
    slam.localization()
    
    # Check if frame buffer is full.
    if len(slam.current_section) < slam.args.section_length:
        if not slam.need_sample:
        # logger.info(f'current_section: {slam.current_section}')
            continue
    ###########################################################################################################
    #                                                   Mapping                                               #
    ###########################################################################################################
    slam.mapping()
    # slam.tracking()
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
    
    # if len(slam.slam_structure.all_frames) > 200:
    # #     slam.slam_structure.filter_mappoints(5)
    # #     slam.slam_structure.save_colmapStyle_data()
    #     break
    
    # if len(slam.slam_structure.key_frames) % 1 == 0:
    #     # slam.slam_structure.filter_mappoints(5)
    #     # slam.slam_structure.save_colmapStyle_data()
    #     # # slam.slam_structure.save_nerfStyle_data()
    #     slam.slam_structure.write_tum_poses(slam.slam_structure.exp_root / "poses_pred.txt")
    #     slam.slam_structure.write_full_tum_poses(slam.slam_structure.exp_root / "f_poses_pred.txt")
        
# for idx in slam.slam_structure.keyframes:
#     slam.localBA.BA.fix_pose(idx, fixed=False)
# slam.localBA.BA.fix_pose(slam.slam_structure.keyframes[0], fixed=True)
# print('Global BA start......')
# slam.localBA.run_ba(opt_iters=20)

# Filter outliers in reconstruction
# NOTE: ONLY FILTER AFTER EVERYTHING IS OPTIMIZED, DOES NOT UPDATE 
slam.slam_structure.filter(min_view_num=2, reprojection_error_threshold=10)
try:
    slam.slam_structure.filter_mappoints(5)
    slam.slam_structure.save_colmapStyle_data()
    slam.slam_structure.save_visualizations()
except:
    print("No visualizations to save.")
# Save data and visualizations
try:
    update_fps=slam.update_counter/slam.total_update_time
    tracking_fps=slam.tracking_counter/slam.total_tracking_time
    mapping_fps=slam.mapping_counter/slam.total_mapping_time
except ZeroDivisionError:
    update_fps=0
    tracking_fps=0
    mapping_fps=0
slam.slam_structure.save_data(dataset, update_fps, tracking_fps, mapping_fps)

print("Done.")