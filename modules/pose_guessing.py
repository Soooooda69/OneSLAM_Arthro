from abc import ABC, abstractmethod
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
        