from abc import ABC, abstractmethod
import g2o
import numpy as np

class PoseGuesserBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, last_poses):
        return np.identity(4)

class PoseGuesserLastPose(PoseGuesserBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, last_poses):
        if len(last_poses) < 1:
            init_poses = np.identity(4)
            # init_poses[2, 3] = 1
            return init_poses
        
        return np.copy(last_poses[-1])

class PoseGuesserConstantVelocity(PoseGuesserBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, last_poses):
        if len(last_poses) < 2:
            return np.identity(4)

        pose_0 = last_poses[-2]
        pose_1 = last_poses[-1]

        try:
            M = pose_1 @ np.linalg.inv(pose_0)
        except:
            breakpoint()
            
        return M @ pose_1

class MotionModel(object):
    def __init__(self, 
            timestamp=None, 
            initial_position=np.zeros(3), 
            initial_orientation=g2o.Quaternion(), 
            initial_covariance=None):

        self.timestamp = timestamp
        self.position = initial_position
        self.orientation = initial_orientation
        self.covariance = initial_covariance    # pose covariance

        self.v_linear = np.zeros(3)    # linear velocity
        self.v_angular_angle = 0
        self.v_angular_axis = np.array([1, 0, 0])

        self.initialized = False
        # damping factor
        self.damp = 0.95

    def current_pose(self):
        '''
        Get the current camera pose.
        '''
        return (g2o.Isometry3d(self.orientation, self.position), 
            self.covariance)

    def predict_pose(self, timestamp):
        '''
        Predict the next camera pose.
        '''
        if not self.initialized:
            return (g2o.Isometry3d(self.orientation, self.position), 
                self.covariance)
        
        dt = timestamp - self.timestamp

        delta_angle = g2o.AngleAxis(
            self.v_angular_angle * dt * self.damp, 
            self.v_angular_axis)
        delta_orientation = g2o.Quaternion(delta_angle)

        position = self.position + self.v_linear * dt * self.damp
        orientation = self.orientation * delta_orientation

        return (g2o.Isometry3d(orientation, position), self.covariance)

    def update_pose(self, timestamp, 
            new_position, new_orientation, new_covariance=None):
        '''
        Update the motion model when given a new camera pose.
        '''
        if self.initialized:
            dt = timestamp - self.timestamp
            assert dt != 0

            v_linear = (new_position - self.position) / dt
            self.v_linear = v_linear

            delta_q = self.orientation.inverse() * new_orientation
            delta_q.normalize()

            delta_angle = g2o.AngleAxis(delta_q)
            angle = delta_angle.angle()
            axis = delta_angle.axis()

            if angle > np.pi:
                axis = axis * -1
                angle = 2 * np.pi - angle

            self.v_angular_axis = axis
            self.v_angular_angle = angle / dt
            
        self.timestamp = timestamp
        self.position = new_position
        self.orientation = new_orientation
        self.covariance = new_covariance
        self.initialized = True
        
class PoseGuesserGTPose(PoseGuesserBase):
    def __init__(self, gt_pose_path) -> None:
        super().__init__()
        self.start_pose = np.identity(4)
        self.gt_pose = {}
        from scipy.spatial.transform import Rotation as Rot
        with open(gt_pose_path, 'r') as f:
            file_lines = f.readlines()
        for idx, line in enumerate(file_lines):
            pose = np.identity(4)
            pose[:3, :3] = Rot.from_quat([float(x) for x in line.split(' ')[4:]]).as_matrix()
            pose[:3, 3] = [float(x) for x in line.split(' ')[1:4]]
            if idx == 0:
                self.start_pose = pose
            # start from identity
            pose = np.linalg.inv(self.start_pose) @ pose
            # normalize the translation
            # pose[:3, 3] /= np.linalg.norm(pose[:3, 3])
            self.gt_pose[int(line.split(' ')[0])] = pose
    def __call__(self, frame_idx):
        return self.gt_pose[frame_idx]
        