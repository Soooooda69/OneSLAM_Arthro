from pathlib import Path
import pickle
import yaml
from scipy.spatial.transform import Rotation as R
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd
from datasets.dataset import ImageDataset
from PIL import Image
import argparse
import random



parser = argparse.ArgumentParser(description='Python file for bringing output of SLAM pipeline into format used by depth network/evaluation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data loading
parser.add_argument("--exp_root", required=True, type=str, help="path to experiment")
parser.add_argument("--process_subset", action='store_true', help="use start_idx/end_idx/image_subsample to define a subset of processed keyframes")
parser.add_argument("--start_idx", default=-1, type=int, help="starting frame index for processing")
parser.add_argument("--end_idx", default=-1, type=int, help="end frame index for processing")


# Parse arguments
args = parser.parse_args()

if args.process_subset:
    assert args.start_idx != -1
    assert args.end_idx != -1

# Load data
exp_root = Path(args.exp_root)
write_root = exp_root / 'depth_training'
write_root.mkdir(parents=True, exist_ok=True)

data_root = exp_root / 'dataset'
dataset = ImageDataset(data_root)

keyframes = []
poses = None # frame_idx: (pose, intrinsics)

points = None # Set of point instances {id: (3d, color)}
pose_point_map = None # Pose to point associations {pose_id: [...(point_id, 2d)]}

pose_images = None

# TODO: Check if there are poses with no valid points

with open(exp_root / 'keyframes.pickle', 'rb') as handle:
    keyframes = pickle.load(handle)

with open(exp_root / 'poses.pickle', 'rb') as handle:
    poses = pickle.load(handle)

with open(exp_root / 'points.pickle', 'rb') as handle:
    points = pickle.load(handle)

with open(exp_root / 'pose_point_map.pickle', 'rb') as handle:
    pose_point_map = pickle.load(handle)

"""
with open(exp_root / 'pose_images.pickle', 'rb') as handle:
    pose_images = pickle.load(handle)
"""

print("Data loaded.")

def write_camera_intrinsics_per_view(destination, dataset):
    reference_sample = dataset[list(dataset.images.keys())[0]]
    intrinsics = reference_sample['intrinsics']
    with open(destination / 'camera_intrinsics_per_view', 'w') as handle:
        for value in intrinsics:
            handle.write(f"{value}\n")

def write_motion(destination, keyframes, poses):
    to_write = dict()

    # Header
    to_write['header'] = dict()
    to_write['header']['seq'] = 0
    to_write['header']['stap'] = 0.
    to_write['header']['frame_id'] = None

    to_write['poses[]'] = dict()
    for i in range(len(keyframes)):
        pose, _ = poses[keyframes[i]]
        trans = pose[:, 3]
        quat = R.from_matrix(pose[:3, :3]).as_quat()

        to_write['poses[]'][f'poses[{i}]'] = dict()
        to_write['poses[]'][f'poses[{i}]']['position'] = dict()
        to_write['poses[]'][f'poses[{i}]']['orientation'] = dict()
        
        to_write['poses[]'][f'poses[{i}]']['position']['x'] = float(trans[0])
        to_write['poses[]'][f'poses[{i}]']['position']['y'] = float(trans[1])
        to_write['poses[]'][f'poses[{i}]']['position']['z'] = float(trans[2])

        to_write['poses[]'][f'poses[{i}]']['orientation']['x'] = float(quat[0])
        to_write['poses[]'][f'poses[{i}]']['orientation']['y'] = float(quat[1])
        to_write['poses[]'][f'poses[{i}]']['orientation']['z'] = float(quat[2])
        to_write['poses[]'][f'poses[{i}]']['orientation']['w'] = float(quat[3])


    with open(destination / 'motion.yaml', 'w') as handle:
        yaml.dump(to_write, handle, default_flow_style=False, sort_keys=False)

def write_selected_indexes(destination, keyframes):
    with open(destination / 'selected_indexes', 'w') as handle:
        for frame in keyframes:
            handle.write(f"{frame}\n")
            
def write_structure(destination, points, visible_point_ids):
    point_cloud = []
    point_colors = []
    for point_id in visible_point_ids:
        point_3d, point_color = points[point_id]
        point_cloud.append(point_3d)
        point_colors.append(point_color)

    #breakpoint()
    print("Point amount: ", len(point_cloud))

    point_cloud = np.array(point_cloud)
    point_colors = (255*np.array(point_colors)).astype(np.uint8)

    point_cloud = pd.DataFrame({'x': point_cloud[:, 0], 'y': point_cloud[:, 1], 'z': point_cloud[:, 2]})
    pynt_cloud = PyntCloud(point_cloud)
    pynt_cloud.to_file(str(destination / 'structure.ply'))

def write_mask(destination, dataset):
    mask = np.repeat(dataset.mask, 3, axis=2)
    im = Image.fromarray(mask)
    im.save(str(destination / 'undistorted_mask.bmp'))
    #im.save(str(destination / 'undistorted_mask.jpg'))


def write_pose_images(destination, keyframes, dataset):
    for frame_idx in keyframes:
        #image_np = np.rollaxis(pose_images[frame_idx]*255, 0, 3)
        im = Image.fromarray(dataset[frame_idx]['image'].astype(np.uint8))
        im.save(str(destination / f'{frame_idx:08}.jpg'))

def write_view_indexes_per_point(destination, keyframes, poses, points, pose_point_map):
    eps = 1e-6
    reprojection_error_threshold = 10
    min_view_num = 3
    max_depth_threshold = 8

    # Collect reprojection_errors to elimnate ouliers
    reprojection_errors = dict()
    min_depths = dict()
    for frame_idx in keyframes:
        pose, intrinsics = poses[frame_idx]
        pose_points = pose_point_map[frame_idx]
        world_to_cam = np.linalg.inv(pose)
        intrinsics_mat = np.identity(3)
        intrinsics_mat[0, 0] = intrinsics[0]
        intrinsics_mat[1, 1] = intrinsics[1]
        intrinsics_mat[0, 2] = intrinsics[2]
        intrinsics_mat[1, 2] = intrinsics[3]

        projection_mat = np.identity(4)[:3]

        #breakpoint()
        for (point_id, point_2d) in pose_points:
            point_3d, _ =  points[point_id]
            point_3d_homogenous = np.append(point_3d, 1)

            point_2d_projected = intrinsics_mat @ projection_mat @ world_to_cam @ point_3d_homogenous
            depth, point_2d_projected = point_2d_projected[2], point_2d_projected[:2]/(eps + point_2d_projected[2])
            
            reprojection_error = np.linalg.norm(point_2d_projected - point_2d)

            if point_id not in reprojection_errors.keys():
                reprojection_errors[point_id] = []
                min_depths[point_id] = 10000000
            
            reprojection_errors[point_id].append((frame_idx, reprojection_error, depth))
            min_depths[point_id] = min(min_depths[point_id], depth)
    
    #breakpoint()


    visible_indices = dict()
    for point_id in reprojection_errors.keys():
        #print(len(reprojection_errors[point_id]), min_depths[point_id])
        if len(reprojection_errors[point_id]) < min_view_num:
            continue
        if min_depths[point_id] > max_depth_threshold:
            continue
            
        for (frame_idx, reprojection_error, depth) in reprojection_errors[point_id]:
            if reprojection_error > reprojection_error_threshold:
                continue
            
            if point_id not in visible_indices.keys():
                visible_indices[point_id] = []
            visible_indices[point_id].append(frame_idx)

            #print(point_id, frame_idx, reprojection_error, depth)
        #breakpoint()
        


    with open(destination / 'view_indexes_per_point', 'w') as handle:
        for point_id in sorted(visible_indices.keys()):
            handle.write(f"{-1}\n")
            for frame_idx in visible_indices[point_id]:
                handle.write(f"{frame_idx}\n")
    
    return sorted(visible_indices.keys())

    
def write_visible_view_indexes(destination, keyframes):
    # TODO: Potentially needs to be adjusted
    with open(destination / 'visible_view_indexes', 'w') as handle:
        for frame in keyframes:
            if args.process_subset and (frame < args.start_idx or frame > args.end_idx):
                continue


            handle.write(f"{frame}\n")


write_selected_indexes(write_root, keyframes)
write_visible_view_indexes(write_root, keyframes)
write_motion(write_root, keyframes, poses)
visible_point_ids = write_view_indexes_per_point(write_root, keyframes, poses, points, pose_point_map)
write_structure(write_root, points, visible_point_ids)
write_mask(write_root, dataset)
write_camera_intrinsics_per_view(write_root, dataset)
write_pose_images(write_root, keyframes, dataset)