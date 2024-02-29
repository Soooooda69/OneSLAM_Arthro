
import numpy as np
from optimizers import BundleAdjuster
import g2o
import torch
from datetime import datetime
from pathlib import Path
import os
import pickle
import shutil
from scipy.spatial.transform import Rotation as R
from datasets.dataset import ImageDataset
from DBoW.R2D2 import R2D2
r2d2 = R2D2()
import cv2
import json
import logging
from itertools import chain
from misc.components import Frame, Camera, KeyFrame, MapPoint, Measurement
logger = logging.getLogger('logfile.txt')

from visualizations import *
import numpy as np

def mat_to_vecs(mat):
    trans_vec = mat[:3, 3]
    rot_vec = R.from_matrix(mat[:3, :3]).as_rotvec()
    return trans_vec.astype(np.float32), rot_vec.astype(np.float32)

def vecs_to_mat(trans_vec, rot_vec):
    mat = np.identity(4)
    mat[:3, 3] = trans_vec
    mat[:3, :3] = R.from_rotvec(rot_vec).as_matrix()
    return mat


class SLAMStructure:
    def __init__(self, dataset, graph, name='', output_folder='./experiments') -> None:
        self.dataset = dataset
        # Internal storage
        self.all_frames = dict() #frame_idx: Frame instance
        self.key_frames = dict() #frame_idx: KeyFrame instance
        self.lc_frames = dict() #frame_idx: localize Frame instance
        self.map_points = dict() #point_id: MapPoint instance

        self.pose_point_edges = dict() # Stores edges for every pose 

        self.last_keyframe = None
        
        self.reference = None # reference keyframe
        self.preceding = None # last keyframe
        self.current = None 

        self.valid_points = set() # Points that have been extracted from a BA run
        self.valid_frames = set() # Poses that are localized in the current map

        # Other Settings
        now = datetime.now()
        self.exp_name = now.strftime("exp_%m%d%Y_%H:%M:%S") if name == '' else name
        self.exp_root = Path(output_folder) / self.exp_name
        self.exp_nerfStyle_root = Path(output_folder) / self.exp_name / 'nerf_data'
        self.exp_colmapStyle_root = Path(output_folder) / self.exp_name / 'colmap_data'
        self.graph = graph
    # Registering new frames

    
    def add_frame(self, frame_idx, pose):
        intrinsic=self.dataset[frame_idx]['intrinsics'].detach().cpu().numpy()
        mask = self.dataset[frame_idx]['mask'].squeeze().detach().cpu().numpy()
        image = self.dataset[frame_idx]['image'].permute(1, 2, 0).detach().cpu().numpy()*255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        timestamp = self.dataset[frame_idx]['timestamp']
        image = image.astype(np.uint8)
        cam = Camera(intrinsic, mask.shape[1], mask.shape[0])
        frame = Frame(frame_idx, pose, mask, image, cam, timestamp)
        self.all_frames[frame_idx] = frame
        return frame
    
    def make_keyframe(self, frame_idx) -> None:
        # Make an existing frame into a new keyframe
        # Assert that frame has been registered and is not already a keyframe
        assert frame_idx not in self.key_frames.keys()
 
        frame = self.all_frames[frame_idx]
        keyframe = frame.to_keyframe()
        self.graph.add_keyframe(keyframe)
        # if self.last_keyframe is not None:
        #     keyframe.update_preceding(self.last_keyframe)
        #     # keyframe.update_preceding(list(self.key_frames.values())[-1])
        self.last_keyframe = keyframe
            
        self.key_frames[frame_idx] = keyframe
    
    def add_map_meas(self, keyframe):
        mappoints = []
        measurements = []
        for mappoint_id, (keypoint, descriptor) in keyframe.feature.keypoints_info.items():
            mappoint = self.map_points[mappoint_id]
            mappoints.append(mappoint)
            
            meas = Measurement(
                Measurement.Source.MAPPING,
                [keypoint],
                [descriptor])
            meas.mappoint = mappoint
            meas.view = keyframe.transform(mappoint.position[...,None])
            measurements.append(meas)
            
        return mappoints, measurements
    
    def add_measurements(self, keyframe, mappoints, measurements):
        for mappoint, measurement in zip(mappoints, measurements):
            self.graph.add_mappoint(mappoint)
            self.graph.add_measurement(keyframe, mappoint, measurement)
            mappoint.increase_measurement_count()  
            
    def create_points(self, keyframe):
        mappoints, measurements = self.add_map_meas(keyframe)
        self.add_measurements(keyframe, mappoints, measurements)
                

    def add_mappoint(self, point_3d, point_color, point_descriptor):
        # Adding a new point
        mapPoint = MapPoint(point_3d, point_descriptor, point_color)
        self.map_points[mapPoint.idx] = mapPoint
        return mapPoint
            
            
    def get_previous_poses(self, n):
        # Return the last n keyframe pose estimates

        # sorted_frames_idx = sorted(self.key_frames.keys())
        # self.key_frames = dict(sorted(self.key_frames.items()))
        # poses = []
        # for frame in self.key_frames.values():
        #     pose = frame.pose
        #     poses.append(pose)
        try:
            return self.last_keyframe.pose
        except:
            return g2o.Isometry3d(np.identity(4))

    def get_point(self, point_id):
        assert point_id in self.points.keys()
        return self.points[point_id]

    # Pose estimation/Mapping functions
    
    def localize_frame(self, frame, update_pose=True, ransac=False):
        # Localize a frame against current map
        
        # TODO: Need scale correction for pose estimation
        
        # assert frame in self.all_frames.keys()
        # print(f"Localizing frame {frame.idx}")
        # current_pose, intrinsics = self.poses[frame_idx]
        current_pose, intrinsics = frame.pose.matrix(), frame.intrinsic
        # pose_points = self.pose_point_map[frame_idx]
        key_points = frame.feature.keypoints_ids
        points_idx_filtered = list(filter(lambda pair: pair in self.valid_points, key_points))
        # points_filtered = frame.feature.keypoints_info[points_idx_filtered]
        points_filtered = [(i, frame.feature.keypoints_info[i][0]) for i in points_idx_filtered]
        # print(f"Localizing frame {frame.idx} with {len(points_filtered)} correspondences.")
        if len(points_filtered) <= 5:
            return current_pose
        
        trans_vec, rot_vec = mat_to_vecs(np.linalg.inv(current_pose))
        
        obj_pts = []
        img_pts = []

        for (point_id, point_2d) in points_filtered:
            # obj_pts.append(self.points[point_id][0])
            obj_pts.append(self.map_points[point_id].position)
            img_pts.append(point_2d)

        obj_pts = np.array(obj_pts)
        img_pts = np.array(img_pts)
        intrinsics_mat = np.identity(3)
        intrinsics_mat[0, 0] = intrinsics[0]
        intrinsics_mat[1, 1] = intrinsics[1]
        intrinsics_mat[0, 2] = intrinsics[2]
        intrinsics_mat[1, 2] = intrinsics[3]
        distCoeffs = np.array([0., 0., 0., 0.])
        if ransac:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(obj_pts, img_pts, intrinsics_mat, distCoeffs, rvec=rot_vec, tvec=trans_vec)
        else:
            retval, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, intrinsics_mat, distCoeffs, rvec=rot_vec, tvec=trans_vec)

        localized_pose = np.linalg.inv(vecs_to_mat(tvec.squeeze(), rvec.squeeze()))
        
        if update_pose:
            frame.update_pose(localized_pose)
        return localized_pose
    
    def filter_points(self, frame):
        local_mappoints = self.graph.get_local_map_v2(
            [self.preceding, self.reference], window_size=12, loop_window_size=8)[0]
        
        # points = []
        # for point in local_mappoints:
            # points.append(point.position)
        can_view = frame.can_view(local_mappoints)
        # print('filter points:', len(local_mappoints), 'can view',can_view.sum(), 
        #     'preceding map:', len(self.preceding.mappoints()),
        #     'reference map:', len(self.reference.mappoints()))
        
        checked = set()
        filtered = []
        for i in np.where(can_view)[0]:
            pt = local_mappoints[i]
            if pt.is_bad():
                continue
            pt.increase_projection_count()
            filtered.append(pt)
            checked.add(pt)
        # print('checked:', len(checked))
        for reference in set([self.preceding, self.reference]):
            for pt in reference.mappoints():  # neglect can_view test
                if pt in checked or pt.is_bad():
                    continue
                pt.increase_projection_count()
                filtered.append(pt)

        return filtered
    

    # def filter(self, max_allowed_distance=None, reprojection_error_threshold=1000000, min_view_num=0):
    #     """
    #     Only retain points and correspondences that are 
    #     - close to some camera pose
    #     - visible on at least a minimum number of views
    #     - have a low reprojection error
    #     """
    #     # TODO: DOES NOT UPDATE BA GRAPH, ONLY DICTS! Might have to construct a seperate BA graph for every optimization.
    #     # TODO: Also filter correspondences in non-keyframes
    #     eps = 1e-6
    #     # Collect minimum cam distances for every point
    #     # Collect reprojection_errors
    #     reprojection_errors = dict()
    #     min_depths = dict()

    #     # for frame_idx in self.keyframes:
    #     for keyframe in self.key_frames.values():
    #         pose, intrinsics = keyframe.pose.matrix(), keyframe.intrinsic
    #         frame_idx = keyframe.idx
    #         # pose_points = self.pose_point_map[frame_idx]
    #         world_to_cam = np.linalg.inv(pose)
    #         intrinsics_mat = np.identity(3)
    #         intrinsics_mat[0, 0] = intrinsics[0]
    #         intrinsics_mat[1, 1] = intrinsics[1]
    #         intrinsics_mat[0, 2] = intrinsics[2]
    #         intrinsics_mat[1, 2] = intrinsics[3]

    #         projection_mat = np.identity(4)[:3]

    #         #breakpoint()
    #         for (point_id, point_2d) in zip(keyframe.feature.keypoints_ids, keyframe.feature.keypoints):
    #         # for (point_id, point_2d) in pose_points:
    #             point_3d = self.map_points[point_id].position
    #             point_3d_homogenous = np.append(point_3d, 1)

    #             point_2d_projected = intrinsics_mat @ projection_mat @ world_to_cam @ point_3d_homogenous
    #             depth, point_2d_projected = point_2d_projected[2], point_2d_projected[:2]/(eps + point_2d_projected[2])
                
    #             reprojection_error = np.linalg.norm(point_2d_projected - point_2d)

    #             if point_id not in reprojection_errors.keys():
    #                 reprojection_errors[point_id] = []
    #                 min_depths[point_id] = 10000000
                
    #             reprojection_errors[point_id].append((frame_idx, reprojection_error, depth))
    #             min_depths[point_id] = min(min_depths[point_id], depth)
        
    #     # If maximum allowed distance not set, take double median distance
    #     if max_allowed_distance is None:
    #         max_allowed_distance = 2 * np.median(np.array(list(min_depths.values())))

    #     # Find out which correspondences are valid
    #     visible_frames = dict() # point_id => [frame_indices]
    #     visible_points = dict() # frame_idx => [point_ids]

    #     for keyframe in self.key_frames.values():
    #         visible_points[keyframe.idx] = [] 

    #     for mappoint in self.map_points.values():
    #         visible_frames[mappoint.idx] = []

    #         if point_id not in reprojection_errors.keys():
    #             continue

    #         if len(reprojection_errors[point_id]) < min_view_num:
    #             continue
    #         if min_depths[point_id] <= 0 or min_depths[point_id] > max_allowed_distance:
    #             continue
            
                
    #         for (frame_idx, reprojection_error, depth) in reprojection_errors[point_id]:
    #             if reprojection_error > reprojection_error_threshold:
    #                 continue
                
    #             visible_frames[point_id].append(frame_idx)
    #             visible_points[frame_idx].append(point_id)
        
    #     filtered_corr_num = 0
    #     # for frame_idx in self.keyframes:
    #     for keyframe in self.key_frames.values():
    #         if keyframe.idx not in self.key_frames.keys():
    #         # if frame_idx not in self.pose_point_map.keys():
    #             continue
            
    #         num_corr_before = len(keyframe.feature.keypoints)
    #         filter_func = (lambda visible_points: (lambda pair: pair[0] in visible_points[frame_idx]))(visible_points)
    #         self.pose_point_map[frame_idx] = list(filter(filter_func, self.pose_point_map[frame_idx]))
    #         keyframe.feature.keypoints = list(filter(filter_func, keyframe.feature.keypoints))
            
    #         num_corr_after = len(self.pose_point_map[frame_idx])

    #         filtered_corr_num += num_corr_before - num_corr_after


    #     print(f"Filtered {filtered_corr_num} correspondences.")

    #     filtered_point_num = 0
    #     for point_id in visible_frames.keys():
    #         if len(visible_frames[point_id]) > 0:
    #             continue
    #         self.points.pop(point_id)
    #         filtered_point_num += 1
        
    #     print(f"Filtered {filtered_point_num} points.")

    def filter_mappoints(self, min_view_num):
        # filter on only visibility
        filtered_point_num = 0
        map_points_copy = self.map_points.copy()
        for mappoint in map_points_copy.values():
            for keyframe in self.key_frames.values():
                if mappoint.idx not in keyframe.feature.keypoints_ids:
                    continue
                mappoint.count['inlier'] += 1

            if mappoint.count['inlier'] < min_view_num:
                del self.map_points[mappoint.idx]
                filtered_point_num += 1
                continue
        print(f"Filtered {filtered_point_num}/{len(self.map_points)} points.")
        
    # Saving data + visualizations
    # TODO: Move to external functions?

    def write_tum_poses(self, path):
        print("Writing tum poses ...")
        with open(path, "w") as tum_file:
            tum_file.write("# frame_index tx ty tz qx qy qz qw\n")

            for keyframe in self.key_frames.values():
                pose = keyframe.pose.matrix()
                t = pose[:3, 3]
                q = R.from_matrix(pose[:3, :3]).as_quat()

                tum_file.write(f'{keyframe.idx} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n')

        print(f'Wrote poses to {path}')
    
    def write_full_tum_poses(self, path):
        print("Writing tum poses ...")
        with open(path, "w") as tum_file:
            tum_file.write("# frame_index tx ty tz qx qy qz qw\n")
            # for frame_index in self.keyframes:
            for frame in self.all_frames.values():
                # pose, _ = self.poses[frame_index]
                pose = frame.pose.matrix()
                t = pose[:3, 3]
                q = R.from_matrix(pose[:3, :3]).as_quat()

                tum_file.write(f'{frame.idx} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n')

        print(f'Wrote poses to {path}')

    def save_data(self, dataset, update_fps, tracking_fps, mapping_fps):
        self.exp_root.mkdir(parents=True, exist_ok=True)
        # Dump class data as pickles (expect image data)
        mappoints_dict = {} #{id: (3d, color, descriptor)}
        for mappoint in self.map_points.values():
            mappoints_dict[mappoint.idx] = (mappoint.position, mappoint.color, mappoint.descriptor)
        with open(self.exp_root / "mappoints.pickle", 'wb') as handle: 
            pickle.dump(mappoints_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        keyframes_dict = {} #{frame_id: [{keypoints_id: (keypoints, descriptors)}, pose, intrinsic]}
        for keyframe in self.key_frames.values():
            keyframes_dict[keyframe.idx] = [keyframe.feature.keypoints_info, keyframe.pose.matrix(), keyframe.intrinsic]
            # cv2.imwrite(str(self.exp_image_root / f"{keyframe.idx}.png"), keyframe.feature.image)
        with open(self.exp_root / "keyframes.pickle", 'wb') as handle:
            pickle.dump(keyframes_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        frames_dict = {} #{frame_id: [{keypoints_id: (keypoints, descriptors)}, pose]}
        for frame in self.all_frames.values():
            frames_dict[frame.idx] = [frame.feature.keypoints_info, frame.pose.matrix()]
        with open(self.exp_root / "frames.pickle", 'wb') as handle:
            pickle.dump(frames_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.write_tum_poses(self.exp_root / "poses_pred.txt")
        self.write_full_tum_poses(self.exp_root / "f_poses_pred.txt")
        
        # with open(self.exp_root / "points.pickle", 'wb') as handle:
        #     pickle.dump(self.points, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(self.exp_root / "pose_point_map.pickle", 'wb') as handle:
        #     pickle.dump(self.pose_point_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

        # Copy dataset
        self.exp_dataset_root = self.exp_root / 'dataset'
        shutil.copytree(dataset.data_root, self.exp_dataset_root)

        # Write info file
        with open(self.exp_root / "info.txt", 'w') as handle:
            handle.write(f"Dataset path {str(dataset.data_root)}\n")
            handle.write(f"Update FPS: {update_fps}\n")
            handle.write(f"Tracking FPS: {tracking_fps}\n")
            handle.write(f"Mapping FPS: {mapping_fps}\n")


        print(f"Written data to {self.exp_root}")

    def save_nerfStyle_data(self):
        self.exp_nerfStyle_root.mkdir(parents=True, exist_ok=True)
        image_root = self.exp_nerfStyle_root / 'train'
        if image_root.exists():
            shutil.rmtree(image_root)
        image_root.mkdir(parents=True, exist_ok=True)
        for keyframe in self.key_frames.values():
            cv2.imwrite(str(image_root / f"{keyframe.idx}.png"), keyframe.feature.image)
        self.save_json_poses()
        plot_points(self.exp_nerfStyle_root / 'points3d.ply', self)
        
    def save_json_poses(self):
        frames = []
        for keyframe in self.key_frames.values():
            transform_matrix = keyframe.transform_matrix.copy()
            translation = transform_matrix[:3, 3]
            transform_matrix[:3, 3] = translation
            frame = {
                "file_path": f"./train/{keyframe.idx}",
                "transform_matrix": transform_matrix.tolist()
            }
            frames.append(frame)
        data = {
            "camera_angle_x": keyframe.hfov,
            "frames": frames
        }
        with open(self.exp_nerfStyle_root /'transforms_train.json', 'w') as f:
            json.dump(data, f, indent=4)
        with open(self.exp_nerfStyle_root /'transforms_test.json', 'w') as f:
            json.dump(data, f, indent=4)
    
    def save_colmapStyle_data(self):
        def save_cameras(sparse_root):
            '''
            # Camera list with one line of data per camera:
            #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            # Number of cameras: 3
            '''
            print('saving cameras.txt to colmap format...')
            intrinsics = self.dataset[1]['intrinsics'].tolist()
            with open(sparse_root / 'cameras.txt', 'w') as f:
                for keyframe in self.key_frames.values():
                    f.write(f"{keyframe.idx} PINHOLE {self.dataset[1]['image'].shape[2]} {self.dataset[1]['image'].shape[1]} {intrinsics[0]} {intrinsics[1]} {intrinsics[2]} {intrinsics[3]}\n")
        
        def save_images(sparse_root):
            '''
            # Image list with two lines of data per image:
            #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            #   POINTS2D[] as (X, Y, POINT3D_ID)
            '''
            print('saving images.txt to colmap format...')

            with open(sparse_root / 'images.txt', 'w') as f:
                for keyframe in self.key_frames.values():
                    # pose = T @ keyframe.pose.matrix()
                    # pose = np.identity(4)
                    # Rot = keyframe.pose.matrix()[:3, :3]
                    # t = keyframe.pose.matrix()[:3, 3]
                    Rot = keyframe.transform_matrix[:3, :3]
                    t = keyframe.transform_matrix[:3, 3]
                    # pose[:3, :3] = Rot
                    # pose[:3, 3] = t
                    # t = pose[:3, 3]
                    q = R.from_matrix(Rot).as_quat()
                    points2d = [[keypoints, keypoints_id] for keypoints_id, (keypoints, _) in keyframe.feature.keypoints_info.items()]
                    f.write(f"{keyframe.idx} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} {keyframe.idx} {keyframe.idx}.png\n")

                    for point2d in points2d:
                        point2d_id = point2d[1]
                        if point2d_id not in self.map_points.keys():
                            f.write(f"{point2d[0][0]} {point2d[0][1]} {-1} ")
                        else:
                            f.write(f"{point2d[0][0]} {point2d[0][1]} {point2d[1]} ")
                    f.write("\n")
                    
        def save_points3d(sparse_root):
            '''
            # 3D point list with one line of data per point:
            #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
            '''
            print('saving points3D.txt to colmap format...')
            with open(sparse_root / 'points3D.txt', 'w') as f:
                for mappoint in self.map_points.values():
                    position = np.array(mappoint.position)
                    color = (255 * np.array(mappoint.color)).astype(np.uint8)
                    f.write(f"{mappoint.idx} {position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]} 0.00 ")
                    for keyframe in self.key_frames.values():
                        if mappoint.idx not in keyframe.feature.keypoints_ids:
                            continue
                        f.write(f"{keyframe.idx} {mappoint.idx} ")
                    f.write("\n")    
                              
        self.exp_colmapStyle_root.mkdir(parents=True, exist_ok=True)
        sparse_root = self.exp_colmapStyle_root /'sparse'/ '0'
        image_root = self.exp_colmapStyle_root / 'images'
        if sparse_root.exists():
            shutil.rmtree(sparse_root)
        sparse_root.mkdir(parents=True, exist_ok=True)
        if image_root.exists():
            shutil.rmtree(image_root)
        image_root.mkdir(parents=True, exist_ok=True)
        for keyframe in self.key_frames.values():
            cv2.imwrite(str(image_root / f"{keyframe.idx}.png"), keyframe.feature.image)
        save_cameras(sparse_root)
        save_images(sparse_root)
        save_points3d(sparse_root)
        plot_points(self.exp_colmapStyle_root / 'points3d.ply', self)
        
    def save_visualizations(self):
        self.visualization_root = self.exp_root / 'vis'
        self.visualization_root.mkdir(parents=True, exist_ok=True)

        plot_points(self.visualization_root / 'points.ply', self)
        plot_and_save_trajectory(self, save_name=str(self.visualization_root / 'trajectory.ply'))


        point_track_root = self.visualization_root / 'point_tracks'
        point_track_root.mkdir(parents=True, exist_ok=True)
        point_track_subsample_factor = 1

        visualize_point_correspondences(point_track_root, self, subsample_factor=point_track_subsample_factor)