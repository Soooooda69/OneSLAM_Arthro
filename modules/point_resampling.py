from abc import ABC, abstractmethod
import numpy as np
from misc.components import Frame
import cv2
import random
from DBoW.R2D2 import R2D2
r2d2 = R2D2()

def sample_from_prob_dist(num_points, prob_dist):
    x = np.linspace(0, int(prob_dist.shape[1]-1), int(prob_dist.shape[1]))
    y = np.linspace(0, int(prob_dist.shape[0]-1), int(prob_dist.shape[0]))

    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    coords = np.stack([xv, yv], axis = 1)

    samples_2d = coords[np.random.choice(len(coords), num_points, replace=False, p=prob_dist.flatten())]
    return samples_2d

def sample_r2d2_features(image, mask, num_points):
    kps, des = r2d2.update_image(image, num_points)
    if des is None:
        return np.array([]), None
    kps = kps[:,:2].astype(int)

    kps_filtered = []
    des_filtered = []

    for i in range(len(kps)):
        if mask[kps[i, 1], kps[i, 0]] == 0:
            continue
            
        kps_filtered.append(kps[i])
        des_filtered.append(des[i])
        
    kps_filtered = np.array(kps_filtered)
    des_filtered = np.array(des_filtered)
        
    return kps_filtered, des_filtered

class PointResamplerBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, current_pose_points, image, depth, intrinsics, slam_structure):
        return current_pose_points, np.array([])

"""
class PointResamplerSimpleFill(PointResamplerBase):
    def __init__(self, min_num_points) -> None:
        super().__init__()
        self.min_num_points = min_num_points

    def __call__(self, current_pose_points, image, depth, intrinsics, mask, slam_structure):
        points_to_sample = max(self.min_num_points - len(current_pose_points), 0)
        prob_dist = mask/mask.sum()

        new_2d_points = sample_from_prob_dist(points_to_sample, prob_dist)

        return current_pose_points, new_2d_points

class PointResamplerMinNewSamples(PointResamplerBase):
    def __init__(self, min_num_points, min_new_points) -> None:
        super().__init__()
        self.min_num_points = min_num_points
        self.min_new_points = min_new_points

    def __call__(self, current_pose_points, image, depth, intrinsics, mask, slam_structure):
        points_to_sample = max(self.min_num_points - len(current_pose_points), self.min_new_points)
        if len(current_pose_points) > 2000:
            points_to_sample = 0
        prob_dist = mask/mask.sum()

        new_2d_points = sample_from_prob_dist(points_to_sample, prob_dist)

        return current_pose_points, new_2d_points

class PointResamplerMinNewSamplesAndForget(PointResamplerBase):
    def __init__(self, min_num_points, min_new_points) -> None:
        super().__init__()
        self.min_num_points = min_num_points
        self.min_new_points = min_new_points

    def __call__(self, current_pose_points, image, depth, intrinsics, mask, slam_structure):
        points_to_sample = max(self.min_num_points - len(current_pose_points), 0)
        offset = 0
        if points_to_sample < self.min_new_points:
            offset = self.min_new_points - points_to_sample
            points_to_sample = self.min_new_points
        
        prob_dist = mask/mask.sum()

        new_2d_points = sample_from_prob_dist(points_to_sample, prob_dist)

        return current_pose_points[offset:], new_2d_points
"""


class PointResamplerUniform(PointResamplerBase):
    def __init__(self, min_num_points, max_num_points, min_new_points) -> None:
        super().__init__()
        self.min_num_points = min_num_points
        self.max_num_points = max_num_points
        self.min_new_points = min_new_points
        

    def __call__(self, current_pose_points, image, depth, intrinsics, mask, slam_structure):
        points_to_sample = max(self.min_num_points - len(current_pose_points), self.min_new_points)
        if len(current_pose_points) > self.max_num_points:
            points_to_sample = 0
        prob_dist = mask/mask.sum()

        new_2d_points = sample_from_prob_dist(points_to_sample, prob_dist)

        return current_pose_points, new_2d_points

class PointResamplerUniformDensity(PointResamplerBase):
    def __init__(self, min_num_points, max_num_points, min_new_points) -> None:
        super().__init__()
        self.min_num_points = min_num_points
        self.max_num_points = max_num_points
        self.min_new_points = min_new_points
        

    def __call__(self, current_pose_points, image, depth, intrinsics, mask, slam_structure):
        points_to_sample = max(self.min_num_points - len(current_pose_points), self.min_new_points)
        if len(current_pose_points) > self.max_num_points:
            points_to_sample = 0

        density = np.ones(mask.shape)

        H, W = mask.shape
        for (point_id, point_2d) in current_pose_points:
            for level in range(2, 10):
                cell_H, cell_W = H//level, W//level
                local_coor = (int(point_2d[0])//cell_W, int(point_2d[1])//cell_H)

                density[local_coor[1]*cell_H:(local_coor[1]+1)*cell_H, local_coor[0]*cell_W:(local_coor[0]+1)*cell_W] += 1


        density = (density.max() + 1) - density
        density *= mask

        prob_dist = density/density.sum()        

        new_2d_points = sample_from_prob_dist(points_to_sample, prob_dist)

        return current_pose_points, new_2d_points


class PointResamplerORB(PointResamplerBase):
    def __init__(self, min_num_points, max_num_points, min_new_points) -> None:
        super().__init__()
        self.min_num_points = min_num_points
        self.max_num_points = max_num_points
        self.min_new_points = min_new_points
        self.orb = cv2.ORB_create()
        

    def __call__(self, current_pose_points, image, depth, intrinsics, mask, slam_structure):
        # image: C x H x W
        # mask: H x W

        points_to_sample = max(self.min_num_points - len(current_pose_points), self.min_new_points)
        if len(current_pose_points) > self.max_num_points:
            points_to_sample = 0
        
        # Prepare image and mask
        image_cv = np.moveaxis(np.copy(image)*255, 0, -1)[:, :, ::-1].astype(np.uint8)
        mask_cv = (np.copy(mask)*255).astype(np.uint8)
        
        # Get orb keypoints
        keypoints= self.orb.detect(image_cv, mask=mask_cv)

        new_2d_points = []

        for keypoint in keypoints:
            # TODO: potential check if keypoint is too close to existing keypoint?
            new_2d_points.append(keypoint.pt)

        new_2d_points = np.array(new_2d_points)

        if len(new_2d_points) > points_to_sample:
            new_2d_points = new_2d_points[:points_to_sample]
        

        if len(new_2d_points) == 0:
            new_2d_points = new_2d_points.reshape((0, 2))

        return current_pose_points, new_2d_points


class PointResamplerSIFT(PointResamplerBase):
    def __init__(self, min_num_points, max_num_points, min_new_points) -> None:
        super().__init__()
        self.min_num_points = min_num_points
        self.max_num_points = max_num_points
        self.min_new_points = min_new_points
        self.sift = cv2.SIFT_create()
        

    def __call__(self, current_pose_points, image, depth, intrinsics, mask, slam_structure):
        # image: C x H x W
        # mask: H x W

        points_to_sample = max(self.min_num_points - len(current_pose_points), self.min_new_points)
        if len(current_pose_points) > self.max_num_points:
            points_to_sample = 0
        
        # Prepare image and mask
        image_cv = np.moveaxis(np.copy(image)*255, 0, -1)[:, :, ::-1].astype(np.uint8)
        mask_cv = (np.copy(mask)*255).astype(np.uint8)
        
        # Get orb keypoints
        keypoints= self.sift.detect(image_cv, mask=mask_cv)

        new_2d_points = []

        for keypoint in keypoints:
            # TODO: potential check if keypoint is too close to existing keypoint?
            new_2d_points.append(keypoint.pt)

        new_2d_points = np.array(new_2d_points)

        if len(new_2d_points) > points_to_sample:
            new_2d_points = new_2d_points[:points_to_sample]
        

        if len(new_2d_points) == 0:
            new_2d_points = new_2d_points.reshape((0, 2))
            
        return current_pose_points, new_2d_points

class PointResamplerAKAZE(PointResamplerBase):
    def __init__(self, min_num_points, max_num_points, min_new_points) -> None:
        super().__init__()
        self.min_num_points = min_num_points
        self.max_num_points = max_num_points
        self.min_new_points = min_new_points
        self.akaze.setDescriptorType(cv2.AKAZE_DESCRIPTOR_MLDB)

    def __call__(self, current_pose_points, image, depth, intrinsics, mask, slam_structure):
        # image: C x H x W
        # mask: H x W

        points_to_sample = max(self.min_num_points - len(current_pose_points), self.min_new_points)
        if len(current_pose_points) > self.max_num_points:
            points_to_sample = 0
        
        # Prepare image and mask
        image_cv = np.moveaxis(np.copy(image)*255, 0, -1)[:, :, ::-1].astype(np.uint8)
        mask_cv = (np.copy(mask)*255).astype(np.uint8)
        
        # Get orb keypoints
        keypoints, _ = self.akaze.detectAndCompute(image_cv, mask=mask_cv)
        new_2d_points = []

        for keypoint in keypoints:
            # TODO: potential check if keypoint is too close to existing keypoint?
            new_2d_points.append(keypoint.pt)

        new_2d_points = np.array(new_2d_points)

        if len(new_2d_points) > points_to_sample:
            new_2d_points = new_2d_points[:points_to_sample]
        

        if len(new_2d_points) == 0:
            new_2d_points = new_2d_points.reshape((0, 2))

        return current_pose_points, new_2d_points

# class PointResamplerR2D2(PointResamplerBase):
#     def __init__(self, min_num_points, max_num_points, min_new_points) -> None:
#         super().__init__()
#         self.min_num_points = min_num_points
#         self.min_new_points = min_new_points
#         self.max_num_points = max_num_points
#         # numbers of image points from r2d2
#         # 2000 to give r2d2 good chances
#         self.num_points = 2000
        
#     def __call__(self, current_pose_points, image, depth, intrinsics, mask, slam_structure):
#         points_to_sample = max(self.min_num_points - len(current_pose_points), self.min_new_points)
#         if len(current_pose_points) >= self.min_num_points:
#             return current_pose_points, None, None
        
#         image = np.transpose(image, (1, 2, 0))*255
#         image = image.astype(np.uint8)


#         new_2d_points, new_points_descriptor = sample_r2d2_features(image, mask, self.num_points)
#         # new_2d_points = new_2d_points[np.random.choice(len(new_2d_points), points_to_sample, replace=False)]
#         new_2d_points = new_2d_points[:points_to_sample]
        
#         return current_pose_points, new_2d_points, new_points_descriptor
    
class PointResamplerR2D2(PointResamplerBase):
    def __init__(self, min_num_points, max_num_points, min_new_points) -> None:
        super().__init__()
        self.min_num_points = min_num_points
        self.min_new_points = min_new_points
        self.max_num_points = max_num_points
        # Init density detection
        self.grid_size = 50
        self.grid_count = None
        self.threshold = None
        # numbers of image points from r2d2
        # 2000 to give r2d2 good chances
        # self.num_points = 2000
        
      
    def __call__(self, frame):
        points_to_sample = max(self.min_num_points - len(frame.feature.keypoints), self.min_new_points)
        # if len(frame.feature.keypoints) >= self.max_num_points:
        #     return frame.feature.keypoints, None, None
        if points_to_sample == 0:
            return frame.feature.keypoints, None, None
        
        current_pose_points = frame.feature.keypoints
        
        if points_to_sample == self.min_num_points:
            new_2d_points, new_points_descriptor = frame.feature.extract(points_to_sample, update=True)
            self.threshold = self.density_count(frame)
            print('Threshold:', self.threshold)
        else:
            # dense detection
            new_2d_points, new_points_descriptor = self.density_sampling(frame, points_to_sample)
            
        # if len(points) > 0:
        #     new_2d_points = np.vstack([new_2d_points, points])
        #     new_points_descriptor = np.vstack([new_points_descriptor, des])
        
        # keep limited number of features
        return current_pose_points, new_2d_points, new_points_descriptor

    def density_sampling(self, frame, points_to_sample):
        image = frame.feature.image.copy()
        h, w = image.shape[:2]
        # Create a grid and count points in each cell
        self.grid_count = np.zeros((h // self.grid_size, w // self.grid_size), dtype=int)
        
        points = frame.feature.keypoints
        for point in points:
            x, y = point
            grid_x = int(x // self.grid_size)
            grid_y = int(y // self.grid_size)
            if grid_x >= self.grid_count.shape[1] or grid_y >= self.grid_count.shape[0]:
                continue
            self.grid_count[grid_y, grid_x] += 1
            
        if self.threshold is None:
            # Define the threshold for low-density cells
            self.threshold = np.mean(self.grid_count) / 2
        
        # Determine the number of points to sample based on the inverse of the density
        new_points = []
        dummy_des = []
        for grid_y in range(self.grid_count.shape[0]):
            for grid_x in range(self.grid_count.shape[1]):
                density = self.grid_count[grid_y, grid_x]
                # print('Density:', density)
                if density < self.threshold:
                    # Inverse proportional sampling: more points in less dense areas
                    num_to_sample = int(((self.threshold - density) / self.threshold) * 5)  # Scale factor 5
                    for _ in range(num_to_sample):
                        new_x = random.randint(grid_x * self.grid_size, (grid_x + 1) * self.grid_size - 1)
                        new_y = random.randint(grid_y * self.grid_size, (grid_y + 1) * self.grid_size - 1)
                        new_points.append([new_x, new_y])
                        dummy_des.append(np.zeros(128))
                        if len(new_points) >= points_to_sample:
                            break
                        
        if len(new_points) > 0:
            return np.vstack(new_points), np.vstack(dummy_des)
        else:
            return np.array([]), None
        
    def density_count(self, frame):
        grid_size = 50
        image = frame.feature.image.copy()
        h, w = image.shape[:2]
        
        # Create a grid and count points in each cell
        grid_count = np.zeros((h // self.grid_size, w // self.grid_size), dtype=int)
    
        for point in frame.feature.keypoints:
            x, y = point
            grid_x = int(x // grid_size)
            grid_y = int(y // grid_size)
            if grid_x >= grid_count.shape[1] or grid_y >= grid_count.shape[0]:
                continue
            grid_count[grid_y, grid_x] += 1
        # Define the threshold for low-density cells
        threshold = np.mean(grid_count)
        return threshold