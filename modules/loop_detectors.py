from abc import ABC, abstractmethod
import numpy as np

from DBoW.R2D2 import R2D2
from modules.point_resampling import sample_r2d2_features
from DBoW.matcher import *
from DBoW.voc_tree import constructTree

class LoopDetectorBase(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.reference_frames = []

    @abstractmethod
    def add_reference_frame(self, frame_idx):
        # Registers a new reference frame
        self.reference_frames.append(frame_idx)

    @abstractmethod
    def search_loops(self, frame_idx, dataset, slam_structure):
        # Returns matching reference frames
        return []


class LoopDetectorBoW(LoopDetectorBase):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.reference_frames = []
        self.reference_kps = []
        self.reference_des = []
        self.to_register = []

        # For r2d2 feature extraction
        self.r2d2_num_points = 1000

        # Tree
        self.K = 5 #classes of cluster
        self.L = 3 #depth of tree

        # Similiarty threshold to detect as reference frame
        self.T = 0.93

        # Minimum distance to count as loop
        self.min_distance = 200

        
        self.tree, self.matcher = None, None
        self.unlearned_ref_frames = 0

        # Construct tree from dataset
        """ Relearning on the fly vs. full learning in the beginning does not seem to change anything
        features = []
        print("Running feature extracation on dataset ...")
        for frame_idx in tqdm(dataset.images.keys()):
            image = dataset[frame_idx]['image'].detach().cpu().numpy()
            mask = dataset[frame_idx]['mask'].squeeze().detach().cpu().numpy()

            image = np.transpose(np.copy(image), (1, 2, 0))*255
            image = image.astype(np.uint8)
            # Extract masked r2d2 keypoints + descriptors 
            kps, des = sample_r2d2_features(image, mask, self.r2d2_num_points)
            features.append(des)
        
        
        N = len(features)
        FEATS = []
        for feats in features:
            FEATS += [np.array(fv, dtype='float32') for fv in feats]
        FEATS = np.vstack(FEATS)

       

        treeArray = constructTree(self.K, self.L, np.vstack(FEATS))
        self.tree = Tree(self.K, self.L, treeArray)
        self.matcher = Matcher(N, features, self.tree)
         """
        


    def add_reference_frame(self, frame_idx):
        # Registers a new reference frame
        self.to_register.append(frame_idx)

    def relearn_tree(self):
        print("Learning new tree ...")
        N = len(self.reference_des)
        FEATS = []
        for feats in self.reference_des:
            FEATS += [np.array(fv, dtype='float32') for fv in feats]
        FEATS = np.vstack(FEATS)

        treeArray = constructTree(self.K, self.L, np.vstack(FEATS))
        self.tree = Tree(self.K, self.L, treeArray)
        self.matcher = Matcher(N, self.reference_des, self.tree)

        for i in range(len(self.reference_frames)):
            self.tree.update_tree(self.reference_frames[i], self.reference_des[i])
        
        self.unlearned_ref_frames = 0

    def search_loops(self, frame_idx, dataset, slam_structure):
        # Registering previously added frames
        
        self.register_keep = []

        for ref_idx in self.to_register:
            if ref_idx >= frame_idx:
                # Ref was not yet processed by tracking
                self.register_keep.append(ref_idx)
                continue

            # Get image + mask
            image = dataset[ref_idx]['image'].detach().cpu().numpy()
            mask = dataset[ref_idx]['mask'].squeeze().detach().cpu().numpy()

            image = np.transpose(np.copy(image), (1, 2, 0))*255
            image = image.astype(np.uint8)
            # Extract masked r2d2 keypoints + descriptors 
            kps, des = sample_r2d2_features(image, mask, self.r2d2_num_points)

            self.reference_kps.append(kps)
            self.reference_des.append(des)
            self.reference_frames.append(ref_idx)

            if self.tree is not None:
                self.tree.update_tree(ref_idx, des)

            self.unlearned_ref_frames += 1

        self.to_register = self.register_keep

        if len(self.reference_frames) == 0:
            return []
        
        
        if self.unlearned_ref_frames > 10 or self.tree is None:
            self.relearn_tree()
        

        # Extract kps and des for current frame
        image = dataset[frame_idx]['image'].detach().cpu().numpy()
        mask = dataset[frame_idx]['mask'].squeeze().detach().cpu().numpy()

        image = np.transpose(np.copy(image), (1, 2, 0))*255
        image = image.astype(np.uint8)
        
        # Extract masked r2d2 keypoints + descriptors 
        kps, des = sample_r2d2_features(image, mask, self.r2d2_num_points)
        if des is  None:
            return []
        
        # Learn new tree
        self.tree.update_tree(frame_idx, des)
        found_loops = []


        for i in range(len(self.reference_frames)):
            
            ref_idx = self.reference_frames[i]
            if abs(ref_idx - frame_idx) < self.min_distance:
                continue

            #matches = bf.match(des.astype(np.float32), self.reference_des[i].astype(np.float32))

            #distances = []
            #import cv2
            #bf = cv2.BFMatcher()
            #breakpoint()
            #for m in matches: distances.append(m.distance)
            #print(f"{ref_idx} => {frame_idx}: {np.mean(distances)}")

            
            if self.matcher.cos_sim(self.tree.transform(frame_idx), self.tree.transform(ref_idx)) < self.T:
                continue

            #breakpoint()
            found_loops.append(ref_idx)
        



        # Returns matching reference frames
        return found_loops
    

