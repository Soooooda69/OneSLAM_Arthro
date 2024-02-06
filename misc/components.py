import numpy as np
import g2o
from collections import defaultdict
from enum import Enum
from misc.feature import ImageFeature
from misc.covisibility import GraphKeyFrame
from misc.covisibility import GraphMapPoint
from misc.covisibility import GraphMeasurement
from threading import Thread, Lock 

class Camera(object):
    def __init__(self, intrinsic, width=640, height=360):
        self.fx = intrinsic[0]
        self.fy = intrinsic[1]
        self.cx = intrinsic[2]
        self.cy = intrinsic[3]

        self.intrinsic_mtx = np.array([
            [self.fx, 0, self.cx], 
            [0, self.fy, self.cy], 
            [0, 0, 1]])

        # self.frustum_near = frustum_near
        # self.frustum_far = frustum_far

        self.width = width
        self.height = height
    

class Frame(object):
    def __init__(self, idx, pose, mask, image, cam, timestamp=None, 
            pose_covariance=np.identity(6)):
        self.idx = idx
        # g2o.Isometry3d
        if isinstance(pose, g2o.Isometry3d):
            self.pose = pose
        else:
            self.pose = g2o.Isometry3d(pose)
        self.orientation = self.pose.orientation()  
        self.position = self.pose.position()
        self.feature = ImageFeature(image, mask, idx)
        self.cam = cam
        self.intrinsic = [self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy] # [fx, fy, cx, cy]
        # self.intrinsic_mtx = np.array([
        #     [intrinsic[0], 0, intrinsic[2]], 
        #     [0, intrinsic[1], intrinsic[3]], 
        #     [0, 0, 1]])
        self.intrinsic_mtx = self.cam.intrinsic_mtx
        self.timestamp = timestamp
        
        self.pose_covariance = pose_covariance

        self.transform_matrix = self.pose.inverse().matrix() # transform from world frame to camera frame
        self.itransform_matrix = self.pose.matrix() # transform from camera frame to world frame
        self.projection_matrix = (
            self.cam.intrinsic_mtx.dot(self.transform_matrix[:3]))  # from world frame to image
    
    def __hash__(self):
        return self.idx
      
    def update_pose(self, pose):
        if isinstance(pose, g2o.Isometry3d):
            self.pose = pose
        else:
            self.pose = g2o.Isometry3d(pose)

        self.orientation = self.pose.orientation()  
        self.position = self.pose.position()

        self.transform_matrix = self.pose.inverse().matrix()[:3]
        self.projection_matrix = (
            self.cam.intrinsic_mtx.dot(self.transform_matrix[:3]))

    def transform(self, points):    # from world coordinates
        '''
        Transform points from world coordinates frame to camera frame.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        '''
        R = self.transform_matrix[:3, :3]
        t = self.transform_matrix[:3, 3]
        return R.dot(points) + t[:, None]

    def itransform(self, points):   # from camera coordinates
        '''
        Transform points from camera coordinates frame to world frame.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        '''
        R = self.itransform_matrix[:3, :3]
        t = self.itransform_matrix[:3, 3]
        return R.dot(points) + t[:, None]
    
    def project(self, points): 
        '''
        Project points from camera frame to image's pixel coordinates.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        Returns:
            Projected pixel coordinates, and respective depth.
        '''
        projection = self.cam.intrinsic_mtx.dot(points / points[-1:])
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
        
    def find_matches(self, source, points, descriptors):
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

        matches = self.feature.find_matches(proj, descriptors)
        matches = dict(matches)

        measurements = []

        for i, j in matches.items():
            # i: projected point local index
            # j: current feature point local index
            meas = Measurement(
                source,
                [self.get_keypoint(j)],
                [self.get_descriptor(j)])
            measurements.append((i, meas))
            self.set_matched(j)
        return matches
    
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
    
    def match_mappoints(self, mappoints, source):
        points = []
        descriptors = []
        for mappoint in mappoints:
            points.append(mappoint.position)
            descriptors.append(mappoint.descriptor)
        matched_measurements = self.find_matches(source, points, descriptors)

        measurements = []
        for i, meas in matched_measurements:
            meas.mappoint = mappoints[i]
            measurements.append(meas)
        return measurements
    
    def to_keyframe(self):
        keyframe = KeyFrame(
            self.idx, self.pose, 
            self.feature.mask, self.feature.image,
            self.cam)
        keyframe.feature = self.feature
        return keyframe
        
class KeyFrame(GraphKeyFrame, Frame):
    # _id = 0
    # _id_lock = Lock()

    def __init__(self, *args, **kwargs):
        GraphKeyFrame.__init__(self)
        Frame.__init__(self, *args, **kwargs)

        # with KeyFrame._id_lock:
        #     self.id = KeyFrame._id
        #     KeyFrame._id += 1

        self.reference_keyframe = None
        self.reference_constraint = None
        self.preceding_keyframe = None
        self.preceding_constraint = None
        self.loop_keyframe = None
        self.loop_constraint = None
        self.fixed = False

    def __hash__(self):
        return self.idx
    
    def update_reference(self, reference=None):
        if reference is not None:
            self.reference_keyframe = reference
        self.reference_constraint = (
            self.reference_keyframe.pose.inverse() * self.pose)

    def update_preceding(self, preceding=None):
        if preceding is not None:
            self.preceding_keyframe = preceding
        self.preceding_constraint = (
            self.preceding_keyframe.pose.inverse() * self.pose)

    def set_loop(self, keyframe, constraint):
        self.loop_keyframe = keyframe
        self.loop_constraint = constraint

    def is_fixed(self):
        return self.fixed

    def set_fixed(self, fixed=True):
        self.fixed = fixed
 
class MapPoint(GraphMapPoint):
    _id = 0
    _id_lock = Lock()

    def __init__(self, position, descriptor,
            color=np.zeros(3), 
            covariance=np.identity(3) * 1e-4):
        super().__init__()

        with MapPoint._id_lock:
            self.id = MapPoint._id
            MapPoint._id += 1

        self.position = position
        # self.normal = normal
        self.descriptor = descriptor
        self.covariance = covariance
        self.color = color
        # self.owner = None

        self.count = defaultdict(int)

    def update_position(self, position):
        self.position = position
    # def update_normal(self, normal):
    #     self.normal = normal
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


class Measurement(GraphMeasurement):
    
    Source = Enum('Measurement.Source', ['TRIANGULATION', 'TRACKING', 'REFIND'])

    def __init__(self, type, source, keypoints, descriptors):
        super().__init__()

        self.source = source
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.view = None    # mappoint's position in current coordinates frame

        self.xy = np.array(self.keypoints[0].pt)
        # if self.is_stereo():
        #     self.xyx = np.array([
        #         *keypoints[0].pt, keypoints[1].pt[0]])

        self.triangulation = (source == self.Source.TRIANGULATION)

    def get_descriptor(self, i=0):
        return self.descriptors[i]
    def get_keypoint(self, i=0):
        return self.keypoints[i]

    def get_descriptors(self):
        return self.descriptors
    def get_keypoints(self):
        return self.keypoints

    def from_triangulation(self):
        return self.triangulation
    def from_tracking(self):
        return self.source == Measurement.Source.TRACKING
    def from_refind(self):
        return self.source == Measurement.Source.REFIND