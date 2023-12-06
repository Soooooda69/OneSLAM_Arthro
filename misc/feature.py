import numpy as np
import cv2

from collections import defaultdict
from numbers import Number

from threading import Thread, Lock 
from queue import Queue
from DBoW.R2D2 import R2D2
r2d2 = R2D2()


class ImageFeature(object):
    def __init__(self, image, mask, idx):
        self.image = image
        self.mask  = mask
        self.idx = idx
        self.height, self.width = image.shape[:2]

        self.keypoints = []      # list of cv2.KeyPoint
        self.descriptors = []    # numpy.ndarray

        self.num_points = 2000
        self.matcher = None

        self.cell_size = 15
        self.distance = 25
        self.matching_neighborhood = 3
        self.neighborhood = (
            self.cell_size * self.matching_neighborhood)

        self._lock = Lock()

    def extract(self):
        keypoints, descriptors = r2d2.update_image(self.image, self.num_points)
        if descriptors is None:
            self.keypoints = np.array([])
            self.descriptors = np.array([])
            self.unmatched = np.array([])
            return
        keypoints = keypoints[:,:2].astype(int)
        kps_filtered = []
        des_filtered = []
        # discard keypoints outside of mask
        for i in range(len(keypoints)):
            if self.mask[keypoints[i, 1], keypoints[i, 0]] == 0:
                continue    
            kps_filtered.append(keypoints[i])
            des_filtered.append(descriptors[i])
             
        self.keypoints = np.array(kps_filtered)
        self.descriptors = np.array(des_filtered)  
        self.unmatched = np.ones(len(self.keypoints), dtype=bool)
    
    def draw_keypoints(self):
        if len(self.image.shape) == 2:
            image = np.repeat(self.image[..., np.newaxis], 3, axis=2)
        else:
            image = self.image
        for keypoint in self.keypoints:
                coord = (int(keypoint[0]), int(keypoint[1]))
                cv2.circle(
                                image,
                                coord,
                                3,
                                color=[25, 255, 20],
                                thickness=-1
                            )
        cv2.imwrite(f'../datasets/temp_data/localize_tracking/{self.idx}.png', image)
        
    def find_matches(self, predictions, descriptors):
        matches = dict()
        distances = defaultdict(lambda: float('inf'))
        for m, query_idx, train_idx in self.matched_by(descriptors):
            if m.distance > min(distances[train_idx], self.distance):
                continue

            pt1 = predictions[query_idx]
            pt2 = self.keypoints[train_idx].pt
            dx = pt1[0] - pt2[0]
            dy = pt1[1] - pt2[1]
            if np.sqrt(dx*dx + dy*dy) > self.neighborhood:
                continue

            matches[train_idx] = query_idx
            distances[train_idx] = m.distance
        matches = [(i, j) for j, i in matches.items()]
        return matches

    def matched_by(self, descriptors):
        with self._lock:
            unmatched_descriptors = self.descriptors[self.unmatched]
            if len(unmatched_descriptors) == 0:
                return []
            lookup = dict(zip(
                range(len(unmatched_descriptors)), 
                np.where(self.unmatched)[0]))

        matches = self.matcher.knnMatch(
            np.array(descriptors), unmatched_descriptors, k=2)
    
        return [(m, m.queryIdx, m.trainIdx) for m in matches]

    def matcher(self):
        # FLANN matcher parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50) # or pass empty dictionary
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # def row_match(self, *args, **kwargs):
    #     return row_match(self.matcher, *args, **kwargs)


    def get_keypoint(self, i):
        return self.keypoints[i]
    
    def get_descriptor(self, i):
        return self.descriptors[i]

    def get_color(self, pt):
        x = int(np.clip(pt[0], 0, self.width-1))
        y = int(np.clip(pt[1], 0, self.height-1))
        color = self.image[y, x]
        if isinstance(color, Number):
            color = np.array([color, color, color])
        return color[::-1] / 255.

    def set_matched(self, i):
        with self._lock:
            self.unmatched[i] = False

    def get_unmatched_keypoints(self):
        keypoints = []
        descriptors = []
        indices = []

        with self._lock:
            for i in np.where(self.unmatched)[0]:
                keypoints.append(self.keypoints[i])
                descriptors.append(self.descriptors[i])
                indices.append(i)

        return keypoints, descriptors, indices



# # TODO: only match points in neighboring rows
# def row_match(matcher, kps1, desps1, kps2, desps2,
#         matching_distance=40, 
#         max_row_distance=2.5, 
#         max_disparity=100):

#     matches = matcher.match(np.array(desps1), np.array(desps2))
#     good = []
#     for m in matches:
#         pt1 = kps1[m.queryIdx].pt
#         pt2 = kps2[m.trainIdx].pt
#         if (m.distance < matching_distance and 
#             abs(pt1[1] - pt2[1]) < max_row_distance and 
#             abs(pt1[0] - pt2[0]) < max_disparity):   # epipolar constraint
#             good.append(m)
#     return good