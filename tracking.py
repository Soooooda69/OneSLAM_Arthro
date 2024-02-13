import torch
import numpy as np
import time
import cv2
import logging
from DBoW.R2D2 import R2D2
from optimizers import BundleAdjuster
from misc.components import Frame, MapPoint, Camera
r2d2 = R2D2()
logger = logging.getLogger('logfile.txt')

# TODO: Rename this to point tracking for more clarity.

# # Helper functions (TODO: Relocate to misc.?)
# def apply_transformation(T, point_cloud):
#     try:
#         for i in range(len(point_cloud)):
#                 point_cloud[i] = T @ point_cloud[i]
            
#         return point_cloud
#     except:
#         breakpoint()

# def unproject(points_2d, depth_image, cam_to_world_pose, intrinsics):
#     # Unproject n 2d points
#     # points: n x 2
#     # depth_image: H x W
#     # cam_pose: 4 x 4 

#     x_d, y_d = points_2d[:, 0], points_2d[:, 1]
#     fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

#     depths = depth_image[y_d.astype(int), x_d.astype(int)]
#     x = ((x_d - cx) * depths / fx)[:, None]
#     y = ((y_d - cy) * depths / fy)[:, None]
#     z = depths[:, None]

#     points_3d = np.stack([x, y, z, np.ones_like(x)],  axis=-1).squeeze(axis=1)

#     points_3d = apply_transformation(cam_to_world_pose, points_3d)

#     return points_3d

# def triangulatePoints(self, pose_0, next_pose, new_points_0, new_points_1):
#         """Triangulates the feature correspondence points with
#         the camera intrinsic matrix, rotation matrix, and translation vector.
#         It creates projection matrices for the triangulation process."""
#         # pose of the next frame
#         R = next_pose[:3 ,:3]
#         t = next_pose[:3 , 3].reshape(3,1)
#         # The canonical matrix (set as the origin)
#         P0 = np.array([[1, 0, 0, 0],
#                        [0, 1, 0, 0],
#                        [0, 0, 1, 0]])
#         P0 = self.K.dot(P0)
#         # Rotated and translated using P0 as the reference point
#         P1 = np.hstack((R, t))
#         P1 = self.K.dot(P1)
#         new_points_0 = cv2.undistortPoints(new_points_0.astype(np.float64), self.K, None)
#         new_points_1 = cv2.undistortPoints(new_points_1.astype(np.float64), self.K, None)
#         # for i in range(curr_points.shape[1]):
#         #     point_3d_homogeneous = cv2.triangulatePoints(P0, P1, curr_points, next_points)
#         #     point_3d = cv2.convertPointsFromHomogeneous(point_3d_homogeneous.T)
#         #     print(point_3d.shape())
#         #     # print(i, point_3d_homogeneous.reshape(-1,4))
#         #     # Minimal motion
#         #     if point_3d_homogeneous[3,:] == 0:
#         #         return None
#         #     else:
#         #         point_3d_homogeneous = point_3d_homogeneous/point_3d_homogeneous[3,:]
#         #         point_3d_homogeneous = pose_0 @ point_3d_homogeneous
#         #         point3d.append(point_3d_homogeneous.reshape(-1,4)[:,:3])
                
#         point_3d_homogeneous = cv2.triangulatePoints(P0, P1, new_points_0, new_points_1)
#         point_3d_homogeneous = pose_0 @ point_3d_homogeneous
#         point_3d = cv2.convertPointsFromHomogeneous(point_3d_homogeneous.T)
#         point_3d = point_3d.reshape((point_3d.shape[0], 3))
#         # print('ppppppppppp',point_3d, t)
#         return point_3d

class Tracking:
    def __init__(self, depth_estimator, point_resampler, tracking_network, target_device, cotracker_window_size, BA, dataset) -> None:
        self.point_resampler = point_resampler
        self.depth_estimator = depth_estimator
        self.tracking_network = tracking_network
        self.target_device = target_device
        self.cotracker_window_size = cotracker_window_size
        self.localBA = BA
        self.optimizer = BundleAdjuster()
        self.dataset = dataset
        self.new_point_ids = []
    
    def refine_pose(self, pose, intrinsics, measurements):
        assert len(measurements) >= 10, (
            'Not enough points')
            
        self.optimizer.clear()
        self.optimizer.add_pose(0, pose, intrinsics, fixed=False)

        for i, m in enumerate(measurements):
            self.optimizer.add_point(i, m.mappoint.position, fixed=True)
            self.optimizer.add_edge(0, i, 0, m.xy)
        self.optimizer.optimize(10)
        return self.optimizer.get_pose(0).matrix()
    
    # Point tracking done
    def process_section(self, section_indices, dataset, slam_structure, 
                    sample_new_points=True, 
                    start_frame=0, 
                    maximum_track_num = None,
                    init = False):        
        # NOTE: This now only does the point tracking and does not add new keyframes or aquires new initial pose estimates 
        #       requires first frame to be present in poses
        # assert section_indices[start_frame] in slam_structure.visited_frames.keys()
        
        # Pad section to atleast five frames
        section_valid = [True for idx in section_indices]
        while len(section_indices) < (self.cotracker_window_size//2) + 1:
            section_indices.append(section_indices[-1])
            section_valid.append(False)

        # Retrieve section data
        samples = []
        for frame_idx in section_indices:
            samples.append(dataset[frame_idx])

        # Point resampling process
        if sample_new_points:
            
            depth_0 = self.depth_estimator(samples[start_frame]['image'], samples[-1]['mask']).squeeze().detach().cpu().numpy()
            # self.depth_estimator.visualize()
                
            frame = slam_structure.all_frames[section_indices[start_frame]]
            
            # local_mappoints = slam_structure.filter_points(frame)
            # measurements = frame.match_mappoints(local_mappoints, Measurement.Source.TRACKING)
            
            # Resample points
            kept_pose_points, new_points_2d, points_descriptor = self.point_resampler(frame, init)
            # print(len(new_points_2d))
            if new_points_2d is not None:
                # Unproject new 2d samples
                new_points_3d = frame.unproject(new_points_2d, depth_0)
                # Add new points and correspondences to datastructure 
                for i in range(len(new_points_3d)):
                    point_3d = new_points_3d[i, :3]
                    point_2d = new_points_2d[i]
                    point_descriptor = points_descriptor[i]
                    # point_color = image_0[:, int(point_2d[1]), int(point_2d[0])]
                    point_color = frame.get_color(point_2d)
                    # mapPoint = MapPoint(point_3d, point_descriptor, point_color)
                    # slam_structure.map_points[mapPoint.id] = mapPoint
                    mapPoint = slam_structure.add_mappoint(point_3d, point_color, point_descriptor)
                    
                    # self.localBA.BA.add_point(mapPoint.id, point_3d)
                    self.new_point_ids.append(mapPoint.id)
                    
                    # update frame
                    frame.feature.keypoints_info[mapPoint.id] = (point_2d, point_descriptor)
                    frame.feature.update_keypoints_info()
            else:
                # If no new points were sampled, add old points to current frame correspondences
                print('no points sampled!')
                frame.feature.keypoints_info = slam_structure.all_frames[section_indices[start_frame]-1].feature.keypoints_info
                frame.feature.update_keypoints_info()
                
        # Obtain currently tracked points on first frame
        # pose_points = slam_structure.get_pose_points(section_indices[start_frame])
        #     keypoints_info = slam_structure.all_frames[section_indices[start_frame]].feature.keypoints_info
        # else:
        #     if section_indices[start_frame] not in slam_structure.lc_frames.keys():
        #         keypoints_info = slam_structure.all_frames[section_indices[start_frame]].feature.keypoints_info
        #     else:
        #         keypoints_info = slam_structure.lc_frames[section_indices[start_frame]].feature.keypoints_info
        keypoints_info = slam_structure.all_frames[section_indices[start_frame]].feature.keypoints_info
    
        if maximum_track_num is not None:
            downsize_kpts_info = dict(list(keypoints_info.items())[:maximum_track_num])
            
        local_point_ids = []
        local_point_descriptors = []
        queries = []
        # If this hits, you ran out of tracked points
        if len(downsize_kpts_info) <= 0:
            if sample_new_points:
                print("Sampling was allowed")
            else:
                print("Sampling was not allowed")
            assert False

        # Generate input data for Co-Tracker
        for point_id, (point2d, des) in downsize_kpts_info.items():
            local_point_ids.append(point_id)
            local_point_descriptors.append(des)
            if point2d[0]< 0 or point2d[1] < 0:
                breakpoint()
            queries.append([start_frame, point2d[0], point2d[1]])
        
        image_seq = torch.cat([sample['image'][None, ...] for sample in samples])[None, ...]
        queries = torch.FloatTensor(queries)[None, ...].to(device=self.target_device)
        # Run Co-Tracker
        # Run tracking network
        # current_time = time.time()
        mask = samples[start_frame]['mask'][None]
        with torch.no_grad():
            pred_tracks, pred_visibility = self.tracking_network(image_seq, 
                                                                queries=queries,
                                                                segm_mask=mask)
            
            # # initialisation
            # self.tracking_network(image_seq,queries=queries,is_first_step=True)
            
            # pred_tracks, pred_visibility = self.tracking_network(image_seq,
            #                                                      queries=queries, 
            #                                                     is_first_step=False,
            #                                                     )
            #print("Cotracker runtime: ", time.time()- current_time)
        # Add new correspondences
        for local_idx in range(start_frame+1, len(section_indices)):
            # Check if frame was duplicated
            if not section_valid[local_idx]:
                continue
            
            # Static frames should not be tracked
            static_threshold = 20
            if np.linalg.norm(pred_tracks[0, local_idx, :].detach().cpu().numpy() - pred_tracks[0, local_idx-1, :].detach().cpu().numpy()) < static_threshold:
                # logger.info(f'No movement from {section_indices[local_idx-1]} to {section_indices[local_idx]}.')
                pred_tracks[0, local_idx, :] = pred_tracks[0, local_idx-1, :]    
                
            # Get frame_idx and mask
            frame_idx = section_indices[local_idx]
            mask = samples[local_idx]['mask'].squeeze().detach().cpu().numpy()
            H, W = mask.shape
            intrinsics = samples[local_idx]['intrinsics'].detach().cpu().numpy()
            image = self.dataset[frame_idx]['image'].permute(1, 2, 0).detach().cpu().numpy()*255
            image = image.astype(np.uint8)
            

            # instantiate frame and store in slam_structure
            # assert frame_idx in slam_structure.all_frames.keys()
            # print('frame_idx:', frame_idx, sample_new_points)
            # cam = Camera(intrinsics)
            # frame = Frame(frame_idx, np.identity(4), mask, image, cam)
            frame = slam_structure.all_frames[frame_idx]
            if not sample_new_points:
                # cam = Camera(intrinsics)
                # frame = Frame(frame_idx, np.identity(4), mask, image, cam)
                slam_structure.lc_frames[frame_idx] = frame 
                        
            # Add new correspondences
            for i in range(len(local_point_ids)):
                # Retrieve point data
                point_id = local_point_ids[i]
                point_descriptor = local_point_descriptors[i]
                tracked_point = pred_tracks[0, local_idx, i].detach().cpu().numpy()

                # Point outside of image boundary goes out of focus
                if tracked_point[0] < 0 or tracked_point[1] < 0 or tracked_point[0] >= W or tracked_point[1] >= H:
                    pred_visibility[0, local_idx, i] = False
                    continue
                
                # Point outside of mask goes out of focus
                if mask[int(tracked_point[1]),  int(tracked_point[0])] == 0:
                    pred_visibility[0, local_idx, i] = False
                    continue
                
                # Check if point has never gone out of focus
                visible = pred_visibility[0, :local_idx+1, i]
                # visible = pred_visibility[0, :len(section_indices), i]
                if not torch.all(visible):
                    continue
                
                # Add actual point
                frame.feature.keypoints_info[point_id] = (tracked_point, point_descriptor)
                frame.feature.update_keypoints_info()
                
        # Visualize tracking
        if sample_new_points:
            for local_idx in range(start_frame, len(section_indices)):
                frame_idx = section_indices[local_idx]
                frame = slam_structure.all_frames[frame_idx]
                frame.feature.draw_keypoints()