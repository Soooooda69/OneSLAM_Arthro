import numpy as np
import g2o
from collections import defaultdict
from misc.feature import ImageFeature
from misc.covisibility import GraphKeyFrame
from misc.covisibility import GraphMapPoint
from misc.covisibility import GraphMeasurement
from threading import Thread, Lock 

class Camera(object):
    def __init__(self, fx, fy, cx, cy, width, height):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.intrinsic = np.array([
            [fx, 0, cx], 
            [0, fy, cy], 
            [0, 0, 1]])

        # self.frustum_near = frustum_near
        # self.frustum_far = frustum_far

        self.width = width
        self.height = height
    

class Frame(object):
    def __init__(self, idx, pose, mask, image, cam, timestamp=None, 
            pose_covariance=np.identity(6)):
        self.idx = idx
        self.pose = pose    # g2o.Isometry3d
        self.orientation = self.pose.orientation()  
        self.position = self.pose.position()
        
        self.feature = ImageFeature(image, mask, idx)
        self.cam = cam
        self.timestamp = timestamp
        
        self.pose_covariance = pose_covariance

        self.transform_matrix = pose.inverse().matrix() # shape: (4, 4)
        self.projection_matrix = (
            self.cam.intrinsic.dot(self.transform_matrix[:3]))  # from world frame to image
        
    def update_pose(self, pose):
        if isinstance(pose, g2o.SE3Quat):
            self.pose = g2o.Isometry3d(pose.orientation(), pose.position())
        else:
            self.pose = pose   
        self.orientation = self.pose.orientation()  
        self.position = self.pose.position()

        self.transform_matrix = self.pose.inverse().matrix()[:3]
        self.projection_matrix = (
            self.cam.intrinsic.dot(self.transform_matrix[:3]))

    def transform(self, points):    # from world coordinates
        '''
        Transform points from world coordinates frame to camera frame.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        '''
        R = self.transform_matrix[:3, :3]
        t = self.transform_matrix[:3, 3]
        return R.dot(points) + t

    def itransform(self, points):   # from camera coordinates
        '''
        Transform points from camera coordinates frame to world frame.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        '''
        R = self.transform_matrix[:3, :3]
        t = self.transform_matrix[:3, 3]
        return R.T.dot(points - t)
    
    def project(self, points): 
        '''
        Project points from camera frame to image's pixel coordinates.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        Returns:
            Projected pixel coordinates, and respective depth.
        '''
        projection = self.cam.intrinsic.dot(points / points[-1:])
        return projection[:2], points[-1]

    def unproject(self, points, depth_image):
        '''
        Unproject points from image's pixel coordinates to camera frame.
        Args:
            points: a point or an array of points, of shape (,2) or (N, 2).
            depth: a scalar or an array of scalars, of shape (1,) or (1, N).
        Returns:
            Unprojected points in camera frame. (N, 3)
        '''
        x_d, y_d = points[:, 0], points[:, 1]
        fx, fy, cx, cy = self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy

        depths = depth_image[y_d.astype(int), x_d.astype(int)]
        x = ((x_d - cx) * depths / fx)[:, None]
        y = ((y_d - cy) * depths / fy)[:, None]
        z = depths[:, None]

        # points_3d = np.stack([x, y, z, np.ones_like(x)],  axis=-1).squeeze(axis=1)
        points_3d = np.stack([x, y, z], axis=1).reshape(3, -1)
        points_3d = self.itransform(points_3d).reshape(-1, 3)
        
        return points_3d
        
    def find_matches(self, points, descriptors):
        '''
        Match to points from world frame.
        Args:
            points: a list/array of points. shape: (N, 3)
            descriptors: a list of feature descriptors. length: N
        Returns:
            List of successfully matched (queryIdx, trainIdx) pairs.
        '''
        points = np.transpose(points)
        proj, _ = self.project(self.transform(points))
        proj = proj.transpose()
        return self.feature.find_matches(proj, descriptors)

    def get_keypoint(self, i):
        return self.feature.get_keypoint(i)
    def get_descriptor(self, i):
        return self.feature.get_descriptor(i)
    def get_color(self, pt):
        return self.feature.get_color(pt)
    def set_matched(self, i):
        self.feature.set_matched(i)
    def get_unmatched_keypoints(self):
        return self.feature.get_unmatched_keypoints()
    
    
class MapPoint(GraphMapPoint):
    _id = 0
    _id_lock = Lock()

    def __init__(self, position, normal, descriptor, 
            color=np.zeros(3), 
            covariance=np.identity(3) * 1e-4):
        super().__init__()

        with MapPoint._id_lock:
            self.id = MapPoint._id
            MapPoint._id += 1

        self.position = position
        self.normal = normal
        self.descriptor = descriptor
        self.covariance = covariance
        self.color = color
        # self.owner = None

        self.count = defaultdict(int)

    def update_position(self, position):
        self.position = position
    def update_normal(self, normal):
        self.normal = normal
    def update_descriptor(self, descriptor):
        self.descriptor = descriptor
    def set_color(self, color):
        self.color = color

    def is_bad(self):
        with self._lock:
            status =  (
                self.count['meas'] == 0
                or (self.count['outlier'] > 20
                    and self.count['outlier'] > self.count['inlier'])
                or (self.count['proj'] > 20
                    and self.count['proj'] > self.count['meas'] * 10))
            return status

    def increase_outlier_count(self):
        with self._lock:
            self.count['outlier'] += 1
    def increase_inlier_count(self):
        with self._lock:
            self.count['inlier'] += 1
    def increase_projection_count(self):
        with self._lock:
            self.count['proj'] += 1
    def increase_measurement_count(self):
        with self._lock:
            self.count['meas'] += 1