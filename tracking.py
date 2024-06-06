import torch
import numpy as np
import time
import cv2
import logging
from DBoW.R2D2 import R2D2
from optimizers import BundleAdjuster
from misc.components import Frame, MapPoint, Camera
from visualizations import *
import statistics
r2d2 = R2D2()
logger = logging.getLogger('logfile.txt')


class Tracking:
    def __init__(self, depth_estimator, point_resampler, tracking_network, target_device, cotracker_window_size, BA, dataset) -> None:
        self.point_resampler = point_resampler
        self.depth_estimator = depth_estimator
        # self.depth_anything = depth_estimator[1]
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
        return self.optimizer.get_pose(0)
    
    def save_allMapPoints(self, slam_structure):
        mapPoint_list = []
        for mappoint in slam_structure.map_points.values():
            mapPoint_list.append(mappoint.position)
        mapPoints = np.vstack(mapPoint_list).transpose() 
        mapPoints = mapPoints.transpose()
        # color = [255, 255, 255]
        # image = current_frame.feature.image
        # for x,y in keypoint_xys:
        #     color.append(image[x.astype(int), y.astype(int)])
        # color = np.vstack(color).astype(np.uint8)
        point_cloud = pd.DataFrame({
            'x': mapPoints[:, 0],
            'y': mapPoints[:, 1],
            'z': mapPoints[:, 2],
            'red': 255,
            'green': 255,
            'blue': 255
        })
        pynt_cloud = PyntCloud(point_cloud)
        pynt_cloud.to_file('maaaaaaaaaaaaaaaaaaaap.ply')
        print('map saved')
        
    # Point tracking done
    def process_section(self, section_indices, dataset, slam_structure, 
                    sample_new_points=True, 
                    start_frame=0, 
                    maximum_track_num = None,
                    mapping=False,
                    initialize=False):        
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
            frame = slam_structure.all_frames[section_indices[start_frame]]
      
            # Resample points
            kept_pose_points, new_points_2d, points_descriptor = self.point_resampler(frame)
            
            if new_points_2d is not None and len(new_points_2d) > 1:
                # Undistort new points
                # ud_points_2d = frame.cam.undistort_points(new_points_2d)
                # new_points_2d = ud_points_2d.squeeze()
                remove_indices = []
                for idx, point_2d in enumerate(new_points_2d):
                    if point_2d[0] < 0 or point_2d[1] < 0 \
                        or point_2d[0] >= frame.cam.width or point_2d[1] >= frame.cam.height:
                        remove_indices.append(idx)
                new_points_2d = np.delete(new_points_2d, remove_indices, axis=0)
                
                print('new points sampled:', len(new_points_2d))
                
                new_points_3d = frame.unproject(new_points_2d[:, [1, 0]], depth_0)
                
                # Add new points and correspondences to datastructure 
                for i in range(len(new_points_3d)):
                    point_3d = new_points_3d[i, :3]
                    point_2d = new_points_2d[i]
                    point_descriptor = points_descriptor[i]
                    point_color = frame.get_color(point_2d)
                    mapPoint = slam_structure.add_mappoint(point_3d, point_color, point_descriptor)
                    
                    # self.localBA.BA.add_point(mapPoint.id, point_3d)
                    self.new_point_ids.append(mapPoint.idx)
                    
                    # update frame
                    frame.feature.keypoints_info[mapPoint.idx] = (point_2d, point_descriptor)
                    frame.feature.update_keypoints_info()
            else:
                # If no new points were sampled, add old points to current frame correspondences
                print('no points sampled!')
                
        # Obtain currently tracked points on first frame
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
                if tracked_point[0] < 0 or tracked_point[1] < 0 \
                    or tracked_point[0] >= W or tracked_point[1] >= H:
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
        if mapping:
            for local_idx in range(start_frame, len(section_indices)):
                frame_idx = section_indices[local_idx]
                frame = slam_structure.all_frames[frame_idx]
                frame.feature.draw_keypoints('../data/temp_data/localize_tracking')
                
                # self.depth_anything(dataset[frame_idx]['image'], dataset[frame_idx]['mask'])
                # self.depth_anything.save_disp(frame_idx)
                # self.depth_anything.visualize(frame_idx)