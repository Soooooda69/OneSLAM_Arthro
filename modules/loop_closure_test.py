import torch
import numpy as np
from scripts.loop_detection import *
from modules.point_resampling import sample_r2d2_features
from DBoW.R2D2 import R2D2
import cv2
r2d2 = R2D2()
import logging
logger = logging.getLogger(__name__)

class LoopClosureR2D2:
	def __init__(self, tracking_network, slam_structure, dataset, start_idx, end_idx):
		self.tracking_network = tracking_network
		self.slam_structure = slam_structure
		self.dataset = dataset
		self.reference_frames = []
		self.img_dict = {}
		self.kps_dict = {}
		self.target_device = 'cuda:0'
		#set threshold for similarity
		# self.T = 0.94 # 300 pts
		self.T = 0.93 # 200 pts
		self.num_points = 200
		#set frames to skip avoid minimum motion
		self.skip_frames = 200
		self.start_idx = start_idx
		self.end_idx = end_idx
		self.downsample_imagedict()
		self.tree, self.matcher = self.r2d2_init(self.img_dict)
		self.vis_save_path = '../datasets/temp_data/loop_detect'
		# self.curr_loop_frame = None
		# remake_vis_dir(self.vis_save_path)
		# self.loops = []
		# self.num_loops = 0
		self.loop_coolDown = 0
		self.loop_mes_list = []
  
	def r2d2_init(self, org_img_dict):

		K = 5 #classes of cluster
		L = 3 #depth of tree
		
		# Cluster the features beforehand, take it as a learning phase.
		if os.path.exists('./DBoW/r2d2_descriptors.pkl'):
			with open('./DBoW/r2d2_descriptors.pkl', 'rb') as file:
				image_descriptors = pickle.load(file)
		else:
			image_descriptors = r2d2.r2d2_features(org_img_dict)
			with open('./DBoW/r2d2_descriptors.pkl', 'wb') as file:
				pickle.dump(image_descriptors, file, protocol=pickle.HIGHEST_PROTOCOL)
		
		# image_descriptors = R2D2.r2d2_features(org_img_dict)
		N = len(image_descriptors)      
		FEATS = []
		
		for feats in image_descriptors:
			FEATS += [np.array(fv, dtype='float32') for fv in feats]
		FEATS = np.vstack(FEATS)
		treeArray = constructTree(K, L, np.vstack(FEATS))
		tree = Tree(K, L, treeArray)
		# tree.build_tree(N, image_descriptors)
		matcher = Matcher(N, image_descriptors, tree)
		return tree, matcher	
 
	def downsample_imagedict(self):
		org_img_dict = self.dataset.acquire_images()
		downsampled_keys = list(org_img_dict.keys())[self.start_idx-1:self.end_idx]
		self.img_dict = {key: org_img_dict[key] for key in downsampled_keys} 

	def add_reference_frame(self, frame_idx):
		self.reference_frames.append(frame_idx)

	def r2d2_loop_detect(self, idx):
		# idx: image frame index
		image = self.img_dict[idx]
		# update the tree with descriptors detected from new image
		kps, des = r2d2.update_image(image, self.num_points)
		self.kps_dict[idx] = kps
		if des is not None:
			self.tree.update_tree(idx, des)
		res = {}
		# skip frames to compare the similarity
		for j in self.tree.imageIDs[:-1]:
			if abs(j - idx) < self.skip_frames:
				continue
			# print('Image {} vs Image {}: {}'.format(idx, j, self.matcher.cos_sim(self.tree.transform(idx), self.tree.transform(j))), end='\r')
			if self.matcher.cos_sim(self.tree.transform(idx), self.tree.transform(j)) >= self.T:
				res[j] = self.matcher.cos_sim(self.tree.transform(idx), self.tree.transform(j))
		if res:
			r = max(res.items(), key=lambda x:x[1])[0]
			print(f"Image {idx} is similar to Image {r} with similarity: ",res[r])
   			# save loop frames to visualize result
			vis_loop_frames(self.img_dict, idx, r, self.kps_dict, save_path=self.vis_save_path)
			return True, [r, idx]
		else:
			return False, []

	def find_loops(self, frame_idx):
		found_loop, loop = self.r2d2_loop_detect(frame_idx)
		if found_loop:
			return loop 

	def find_motion(self, loop):
		# loop_mes_list = []
		pose = np.identity(4)
		intrinsics = self.dataset[loop[0]]['intrinsics']
		K = np.identity(3)
		K[0,0] = intrinsics[0]
		K[0,1] = intrinsics[1]
		K[0,2] = intrinsics[2]
		K[1,2] = intrinsics[3]
		num_points = 2000
  
		# for [ref_id, query_id] in loops:
		ref_id = loop[0]
		query_id = loop[1]
  
		# ref_img = self.dataset[ref_id]['image'].permute(1, 2, 0).detach().cpu().numpy()*255
		# ref_keypoints, ref_des = r2d2.update_image(ref_img, num_points)
  		# # TODO: use the old keypoints and descriptors for reference image
		# # ref_keypoints = [point_info[1] for point_info in self.slam_structure.pose_point_map[ref_id]]
		# # ref_des = [point_info[2] for point_info in self.slam_structure.pose_point_map[ref_id]]
  
		# query_img = self.dataset[query_id]['image'].permute(1, 2, 0).detach().cpu().numpy()*255
		# query_keypoints, query_des = r2d2.update_image(query_img, num_points)
		# # FLANN parameters
		# FLANN_INDEX_KDTREE = 1
		# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		# search_params = dict(checks=50) # or pass empty dictionary
		# flann = cv2.FlannBasedMatcher(index_params,search_params)
		# ref_des = np.array(ref_des).astype(np.float32)
		# query_des = np.array(query_des).astype(np.float32)
		# matches = flann.knnMatch(ref_des, query_des, k=2)

		# good_matches = []
		# for m, n in matches:
		# 	if m.distance < 0.75 * n.distance:
		# 		good_matches.append(m)

		# keypoints1 = ref_keypoints[:,:2]
		# keypoints2 = query_keypoints[:,:2]
		# # keypoints1 = [cv2.KeyPoint(x, y, 1) for [x, y] in ref_keypoints]
		# keypoints1 = [cv2.KeyPoint(x, y, 1) for x, y in keypoints1]
		# keypoints2 = [cv2.KeyPoint(x, y, 1) for x, y in keypoints2]
		# # print('kp1,kp2',len(keypoints1), len(keypoints2))
		# ref_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
		# query_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
		# # print('ref,que',ref_keypoints1.shape, query_keypoints2.shape)
		# # Minimal motion between frames
		# if np.linalg.norm(query_keypoints2 - ref_keypoints1) < 20:
		# 	print('Minimal motion.', np.linalg.norm(query_keypoints2 - ref_keypoints1))
		# 	pose = np.identity(4)
		# else:
		# 	E, mask = cv2.findEssentialMat(query_keypoints2, ref_keypoints1, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		# 	_, R, t, _ = cv2.recoverPose(E, query_keypoints2, ref_keypoints1, cameraMatrix=K)
		# 	pose[:3, :3] = R
		# 	pose[:3, 3] = t.reshape(3,)*0.001
		# ##
		# pose = np.identity(4)
		# loop_mes_list.append([ref_id, query_id, pose])
	# return loop_mes_list
		# breakpoint()
		return [ref_id, query_id, pose]