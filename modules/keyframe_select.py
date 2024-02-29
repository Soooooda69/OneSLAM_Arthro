from abc import ABC, abstractmethod
import numpy as np

class KeyframeSelect(ABC):
    def __init__(self, dataset, depth_estimator, slam_structure, localBA) -> None:
        super().__init__()
        self.dataset = dataset
        self.depth_estimator = depth_estimator
        self.slam_structure = slam_structure
        self.localBA = localBA
        self.keyframe_cooldown = 10
        self.new_keyframe_counter = 0
        self.new_keyframes = []
        self.make_keyframe = False
        
    @abstractmethod
    def decide_keyframe(self):
        # Returns matching reference frames
        return []
    
    def setKeyframe(self, idx, isLoopKeyframe=False):

        self.slam_structure.make_keyframe(idx)
        # self.localBA.set_frame_data(idx, fixed=False)
        self.new_keyframe_counter += 1
        self.new_keyframes.append(idx)
        # self.slam_structure.all_frames[idx].to_keyframe()
        print('keyframe:', idx)
        
    def resetNewKeyframeCounter(self):
        self.new_keyframe_counter = 0
        self.new_keyframes = []
       
    def run(self, idx):
        self.decide_keyframe()
        if self.make_keyframe:
            # print('keyframe:', idx)
            self.setKeyframe(idx)
            self.make_keyframe = False
    
class KeyframeSelectSubsample(KeyframeSelect):
    def __init__(self, dataset, depth_estimator, slam_structure, localBA, keyframe_subsample) -> None:
        super().__init__(dataset, depth_estimator, slam_structure, localBA)
        self.keyframe_subsample =  keyframe_subsample
        
    def decide_keyframe(self):
        self.keyframe_cooldown -= 1
        if self.keyframe_cooldown <= 0:
            self.keyframe_cooldown = self.keyframe_subsample
            self.make_keyframe = True

    
class KeyframeSelectFeature(KeyframeSelect):
    def __init__(self, dataset, depth_estimator, slam_structure, localBA) -> None:
        super().__init__(dataset, depth_estimator, slam_structure, localBA)
        self.cooldown_num = 3
        self.similar_threshold = 0.90
        
    def decide_keyframe(self, idx): 
        self.keyframe_cooldown -= 1
        current_frame = self.slam_structure.all_frames[idx]

        # Compare similarity of tracked points between last keyframe and current frame
        last_keypoints_ids = self.slam_structure.last_keyframe.feature.keypoints_ids
        last_point_ids = set()
        for point_id in last_keypoints_ids: last_point_ids.add(point_id)

        # current_pose_points = self.slam_structure.pose_point_map[idx]
        current_point_ids = current_frame.feature.keypoints_ids
        tracked_point_ids = set()
        for point_id in current_point_ids: tracked_point_ids.add(point_id)
        
        #TODO: also check the parallax between the keyframes
        current_pose = current_frame.pose.matrix()
        last_pose = self.slam_structure.last_keyframe.pose.matrix()
        parallax = np.linalg.norm(current_pose[:3, 3] - last_pose[:3, 3])
        # print('parallax:', parallax)
        if self.jaccard_similarity(tracked_point_ids, last_point_ids) < self.similar_threshold\
            and self.keyframe_cooldown <= 0:
            # print(self.keyframe_cooldown)
            self.make_keyframe = True
            
        if self.make_keyframe:
            self.keyframe_cooldown = self.cooldown_num
            
    # def decide_keyframe(self, idx):
    #     self.keyframe_cooldown -= 1
    #     if idx == 1:
    #         # Check if last keyframe was old
    #         self.make_keyframe = True
    #     else:
    #         last_keypoints_ids = self.slam_structure.last_keyframe.feature.keypoints_ids
    #         last_point_ids = set()
    #         for point_id in last_keypoints_ids: last_point_ids.add(point_id)

    #         # current_pose_points = self.slam_structure.pose_point_map[idx]
    #         current_point_ids = self.slam_structure.all_frames[idx].feature.keypoints_ids
    #         tracked_point_ids = set()
    #         for point_id in current_point_ids: tracked_point_ids.add(point_id)
            
    #         if self.jaccard_similarity(tracked_point_ids, last_point_ids) < self.similar_threshold and self.keyframe_cooldown <= 0:
    #             self.make_keyframe = True

    #     if self.make_keyframe:
    #         self.keyframe_cooldown = self.cooldown_num
    
    def jaccard_similarity(self, set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0
    
    def run(self, idx):
        if idx in self.slam_structure.key_frames.keys():
            return
        self.decide_keyframe(idx)
        if self.make_keyframe:
            # print('keyframe:', idx)
            self.setKeyframe(idx)
            self.make_keyframe = False
        # self.setKeyframe(idx)