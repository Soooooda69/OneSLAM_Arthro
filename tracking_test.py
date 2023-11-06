import torch
import numpy as np

# Helper functions (TODO: Relocate to misc.?)
def apply_transformation(T, point_cloud):
    try:
        for i in range(len(point_cloud)):
                point_cloud[i] = T @ point_cloud[i]
            
        return point_cloud
    except:
        breakpoint()

def unproject(points_2d, depth_image, cam_to_world_pose, intrinsics):
    # Unproject n 2d points
    # points: n x 2
    # depth_image: H x W
    # cam_pose: 4 x 4 

    x_d, y_d = points_2d[:, 0], points_2d[:, 1]
    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

    depths = depth_image[y_d.astype(int), x_d.astype(int)]
    x = ((x_d - cx) * depths / fx)[:, None]
    y = ((y_d - cy) * depths / fy)[:, None]
    z = depths[:, None]

    points_3d = np.stack([x, y, z, np.ones_like(x)],  axis=-1).squeeze(axis=1)

    points_3d = apply_transformation(cam_to_world_pose, points_3d)

    return points_3d



class Tracking:
    def __init__(self, pose_guesser, depth_estimator, point_resampler, tracking_network, target_device) -> None:
        self.point_resampler = point_resampler
        self.depth_estimator = depth_estimator # TODO: Implement
        self.pose_guesser = pose_guesser
        self.tracking_network = tracking_network
        self.target_device = target_device

    def process_section(self, section_indices, section_keyframes, dataset, slam_structure):        
        assert section_indices[0] in section_keyframes
        
        # Retrieve section data and add any potential new keyframes
        samples = []
        image_0, depth_0, pose_0, intrinsics_0, mask_0 = None, None, None, None, None
        for frame_idx in section_indices:
            samples.append(dataset[frame_idx])
            image = samples[-1]['image'].detach().cpu().numpy()
            depth = self.depth_estimator(samples[-1]['image'], samples[-1]['mask']).squeeze().detach().cpu().numpy()
            intrinsics = samples[-1]['intrinsics'].detach().cpu().numpy()
            mask = samples[-1]['mask'].squeeze().detach().cpu().numpy()
            mask[depth < 1e-6] = 0
            
            if frame_idx not in section_keyframes:
                continue

            if frame_idx not in slam_structure.keyframes:
                intial_pose_guess = self.pose_guesser(slam_structure.keyframes, slam_structure.poses)
                slam_structure.add_keypose(frame_idx, intial_pose_guess, intrinsics, image, depth, mask)

            pose, _ = slam_structure.poses[frame_idx]

            if frame_idx == section_indices[0]:
                image_0, depth_0, pose_0, intrinsics_0, mask_0 = image, depth, pose, intrinsics, mask
            
        # Get current points
        current_pose_points = slam_structure.get_pose_points(section_indices[0])

        # Resample points
        kept_pose_points, new_points_2d = self.point_resampler(current_pose_points, image_0, depth_0, intrinsics_0, mask_0, slam_structure)
        
        
        # Unproject new 2d samples
        new_points_3d = unproject(new_points_2d, depth_0, pose_0, intrinsics_0)

        # Add new point and correspondence to datastructure 
        for i in range(len(new_points_3d)):
            
            point_3d = new_points_3d[i, :3]
            point_2d = new_points_2d[i]

            point_color = image_0[:, int(point_2d[1]), int(point_2d[0])]

            point_id = slam_structure.add_point(point_3d, point_color)
            slam_structure.add_pose_point(section_indices[0], point_id, point_2d)

            kept_pose_points.append((point_id, point_2d))

        # Run tracking network
        local_point_ids = []
        queries = []

        for (point_id, point2d) in kept_pose_points:
            local_point_ids.append(point_id)
            if point2d[0]< 0 or  point2d[1] < 0:
                breakpoint()
            queries.append([0, point2d[0], point2d[1]])
        
        image_seq = torch.cat([sample['image'][None, ...] for sample in samples])[None, ...]
        queries = torch.FloatTensor(queries)[None, ...].to(device=self.target_device)

        pred_tracks, pred_visibility = self.tracking_network(image_seq, queries=queries)


        # TODO: Get mask explicitly. 

        H, W = mask.shape
        # Add new correspondences


        for local_idx in range(1, len(section_indices)):
            frame_idx = section_indices[local_idx]

            for i in range(len(local_point_ids)):
                point_id = local_point_ids[i]
                tracked_point = pred_tracks[0, local_idx, i].detach().cpu().numpy()

                if tracked_point[0] < 0 or tracked_point[1] < 0 or tracked_point[0] >= W or tracked_point[1] >= H:
                    pred_visibility[0, local_idx, i] = False
                    continue

                if mask[int(tracked_point[1]),  int(tracked_point[0])] == 0:
                    pred_visibility[0, local_idx, i] = False
                    continue

                visible = pred_visibility[0, :local_idx+1, i]
                if torch.all(visible):
                    slam_structure.add_pose_point(frame_idx, point_id, tracked_point)

        # Tracking done