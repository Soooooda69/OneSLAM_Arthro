import numpy as np
import open3d as o3d
import pytransform3d.transformations as pt
import copy
from pyntcloud import PyntCloud
import pandas as pd
from PIL import Image
from pathlib import Path

def plot_and_save_trajectory(
    slam_structure,
    axlen=1,
    subsampling_factor=1,
    save_name="trajectory.ply",
    draw_connections=True,
    connection_color=[1, 0, 0],
    show=False,
):
    pose_array = []
    for keyframe in slam_structure.key_frames.values(): 
        print(f'{keyframe.idx:08}:', keyframe.pose.matrix()[:3, 3])
        pose_array.append(keyframe.pose.matrix())
    
    poses = pose_array

    tm = []  # Temporal transformations
    transformation_matrices = np.empty((len(poses), 4, 4))
    points = []
    for j, cam_pose in enumerate(poses):
        if j % subsampling_factor != 0:
            continue
        # rot = np.transpose(cam_pose[0:3, 0:3])
        rot = cam_pose[0:3, 0:3]
        transl = cam_pose[:3, 3]
        # transl = np.matmul(-np.transpose(rot), cam_pose[:3, 3])
        points.append(transl)
        tm.append(pt.transform_from(R=rot, p=transl))
    transformation_matrices = np.asarray(tm)
    trajectory = None
    for i, pose in enumerate(transformation_matrices):
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axlen)
        mesh_frame.transform(pose)
        if i == 0:
            trajectory = copy.deepcopy(mesh_frame)
        else:
            trajectory += copy.deepcopy(mesh_frame)
    if draw_connections:
        lines = []
        for i in range(len(points) - 1):
            lines.append([i, i + 1])
        colors = [connection_color for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
    if show:
        o3d.visualization.draw_geometries([trajectory, line_set])
    o3d.io.write_triangle_mesh(save_name, trajectory)

def plot_points(filepath, slam_structure):
    # Count number of point occurrences to avoid untriangulated points
    scale = 10
    counts = dict()
    for keyframe in slam_structure.key_frames.values():
        for point_id in keyframe.feature.keypoints_ids:
            if point_id not in counts:
                counts[point_id] = 0
            counts[point_id] += 1

    point_cloud = []
    point_colors = []
    point_normals = []
    for mappoint in slam_structure.map_points.values():
        point_id = mappoint.idx
        # if point_id not in counts or counts[point_id] <= 3:
        #     continue
        point_3d, point_color = mappoint.position, mappoint.color
        point_cloud.append(point_3d)
        point_colors.append(point_color)

    point_cloud = np.array(point_cloud)
    point_colors = (255 * np.array(point_colors)).astype(np.uint8)
    point_normals = np.zeros_like(point_cloud)

    # Normalize the point cloud
    point_cloud_mean = np.mean(point_cloud, axis=0)
    point_cloud_std = np.std(point_cloud, axis=0)
    point_cloud_normalized = ((point_cloud - point_cloud_mean) / point_cloud_std) * scale

    point_cloud_normalized = pd.DataFrame({
        'x': point_cloud_normalized[:, 0],
        'y': point_cloud_normalized[:, 1],
        'z': point_cloud_normalized[:, 2],
        'red': point_colors[:, 0],
        'green': point_colors[:, 1],
        'blue': point_colors[:, 2],
        'nx': point_normals[:, 0],
        'ny': point_normals[:, 1],
        'nz': point_normals[:, 2]
    })

    pynt_cloud = PyntCloud(point_cloud_normalized)
    pynt_cloud.to_file(str(filepath))
    return point_cloud_mean, point_cloud_std, scale
    
def visualize_point_correspondences(path, slam_structure, subsample_factor = 1):
    def get_color(id):
        import colorsys
        PHI = (1 + np.sqrt(5))/2
        n = id * PHI - np.floor(id * PHI) 
        hue = np.floor(n * 256)

        return np.array(colorsys.hsv_to_rgb(hue/360.0, 1, 1))

    frames = []
    indices = []

    for keyframe in slam_structure.key_frames.values():
        indices.append(keyframe.idx)
        img = np.copy(keyframe.feature.image) # x,y,c
        img = np.transpose(img, (2, 0, 1)) # c,x,y
        # point_info = slam_structure.pose_point_map[frame_idx]

        for (point_id, point_2d) in zip(keyframe.feature.keypoints_ids, keyframe.feature.keypoints):
            if point_id%subsample_factor != 0:
                continue
            point_2d_org_x = min(max(4, int(point_2d[0])), img.shape[2]-5)
            point_2d_org_y = min(max(4, int(point_2d[1])), img.shape[1]-5)

            color = get_color(point_id)

            size = 3

            for i in range(-size, size+1):
                for j in range(-size, size+1):
                    img[:, point_2d_org_y+j, point_2d_org_x+i] = color
            
        # frames.append(np.rollaxis(img*255, 0, 3).astype(np.uint8))
        frames.append(np.rollaxis(img, 0, 3).astype(np.uint8))
        
    for i in range(len(frames)):
        im = Image.fromarray(frames[i])
        im.save(str(path / (f'{indices[i]:06}'+".png")))