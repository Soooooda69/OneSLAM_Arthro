from abc import ABC, abstractmethod

class KeyframeSelect(ABC):
    def __init__(self, dataset, depth_estimator, slam_structure, localBA) -> None:
        super().__init__()
        self.dataset = dataset
        self.depth_estimator = depth_estimator
        self.slam_structure = slam_structure
        self.localBA = localBA
        self.keyframe_cooldown = 0
        self.new_keyframe_counter = 0
        self.new_keyframes = []
        self.make_keyframe = False
        
    @abstractmethod
    def decide_keyframe(self):
        # Returns matching reference frames
        return []
    
    def setKeyframe(self, idx):
        # Update keyframe in slam_structure and localBA
        image = self.dataset[idx]['image'].detach().cpu().numpy()
        depth = self.depth_estimator(self.dataset[idx]['image'], self.dataset[idx]['mask']).squeeze().detach().cpu().numpy()
        mask = self.dataset[idx]['mask'].squeeze().detach().cpu().numpy()
        mask[depth < 1e-6] = 0
        self.slam_structure.make_keyframe(idx, image, depth, mask)   
        self.localBA.set_frame_data(idx, fixed=False)
        self.new_keyframe_counter += 1
        self.new_keyframes.append(idx)
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
        self.feature_keyframe_cooldown = 400
        self.similar_threshold = 0.90
        
    def decide_keyframe(self, idx):
        self.keyframe_cooldown -= 1
        if self.keyframe_cooldown <= 0:
            # Check if last keyframe was old
            self.make_keyframe = True
        else:
            last_keyframe = self.slam_structure.keyframes[-1]
            last_pose_points =  self.slam_structure.pose_point_map[last_keyframe]
            last_point_ids = set()
            for (point_id, point_2d) in last_pose_points: last_point_ids.add(point_id)

            current_pose_points = self.slam_structure.pose_point_map[idx]

            tracked_point_ids = set()
            for (point_id, point_2d) in current_pose_points:
                if point_id in last_point_ids: tracked_point_ids.add(point_id)
            
            # print(f'jaccard_similarity between{idx}, {last_keyframe}:',self.jaccard_similarity(tracked_point_ids, last_point_ids))
            # if len(tracked_point_ids)/len(last_point_ids) < 0.8:
            if self.jaccard_similarity(tracked_point_ids, last_point_ids) < self.similar_threshold:
                self.make_keyframe = True

        if self.make_keyframe:
            self.keyframe_cooldown = self.feature_keyframe_cooldown
    
    def jaccard_similarity(self, set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0
    
    def run(self, idx):
        self.decide_keyframe(idx)
        if self.make_keyframe:
            # print('keyframe:', idx)
            self.setKeyframe(idx)
            self.make_keyframe = False
