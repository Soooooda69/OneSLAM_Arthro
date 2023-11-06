import torch
import numpy as np
from scripts.loop_detection import *
from DBoW.R2D2 import R2D2
import cv2
r2d2 = R2D2()
 
class LoopClosureBasic:
	def __init__(self, tracking_network, depth_estimator, target_device):
		self.tracking_network = tracking_network
		self.depth_estimator = depth_estimator
		self.target_device = target_device
		self.reference_frames = []
	
	def add_reference_frame(self, frame_idx):
		self.reference_frames.append(frame_idx)
	
	def check_and_close_loop(self, frame_idx, reference_idx, dataset, slam_structure):
		def get_color(id):
			import colorsys
			PHI = (1 + np.sqrt(5))/2
			n = id * PHI - np.floor(id * PHI) 
			hue = np.floor(n * 256)

			return np.array(colorsys.hsv_to_rgb(hue/360.0, 1, 1))
		
		if abs(frame_idx - reference_idx) < 200:
			return False, []
		
	
		sample_ref = dataset[reference_idx]
		sample_cur = dataset[frame_idx]
		
		image_ref = sample_ref['image'].detach().cpu().numpy()
		depth_ref = self.depth_estimator(sample_ref['image'], sample_ref['mask']).squeeze().detach().cpu().numpy()
		intrinsics_ref = sample_ref['intrinsics'].detach().cpu().numpy()
		mask_ref = sample_ref['mask'].squeeze().detach().cpu().numpy()
		mask_ref[depth_ref < 1e-6] = 0




		pose_points = slam_structure.get_pose_points(reference_idx)

		# Run tracking network
		local_point_ids = []
		queries = []

		for (point_id, point2d) in pose_points:
			local_point_ids.append(point_id)
			if point2d[0]< 0 or  point2d[1] < 0:
				breakpoint()
			queries.append([0, point2d[0], point2d[1]])
		
		image_seq = torch.cat([sample['image'][None, ...] for sample in [ sample_ref, sample_cur, sample_cur, sample_cur, sample_cur, sample_cur, sample_cur, sample_cur]])[None, ...]
		queries = torch.FloatTensor(queries)[None, ...].to(device=self.target_device)

		pred_tracks, pred_visibility = self.tracking_network(image_seq, queries=queries)

		# TODO: Check if loop occured and add correspondences


		visible_point_ratio = pred_visibility[0, -1, :].long().sum() / len(pred_visibility[0, -1, :]) 
		#breakpoint()
		print(f"RATIO {visible_point_ratio}")
		print((pred_tracks[0, 0]-pred_tracks[0, -1]).abs().mean(dim=0))
		
		from torchvision.utils import save_image
		#save_image(image_seq[0, -1], "tgt.png")
		
		# Initial check for loop closure by checking co-tracker success
		if visible_point_ratio < 0.5:
			# Too little points visible, likely to gone out of focus
			return False, []
		
		
		if visible_point_ratio >= 0.98:
			# Too many points visible, unlikely in case of loop closure 
			return False, []
		

		if (pred_tracks[0, 0]-pred_tracks[0, -1])[pred_visibility[0, -1, :]].abs().mean(dim=0).min() < 10 or (pred_tracks[0, 0]-pred_tracks[0, -1])[pred_visibility[0, -1, :]].abs().mean(dim=0).max() < 20:
			# Points did not significantly move.
			return False, []
		
		local_point_ids_reverse = []
		for i in range(len(local_point_ids)):
			if not pred_visibility[0, -1, i]:
				continue
			local_point_ids_reverse.append(local_point_ids[i])

		# TODO: Reverse run of Co-Tracker to confirm tracked points
		image_seq_reverse = torch.cat([sample['image'][None, ...] for sample in [ sample_cur, sample_ref, sample_ref, sample_ref, sample_ref, sample_ref, sample_ref, sample_ref]])[None, ...]
		queries_reverse = torch.clone(queries)

		queries_reverse[:, pred_visibility[0, -1, :], 1:] = pred_tracks[0, -1][pred_visibility[0, -1, :], :]
		queries_reverse = queries_reverse[:, pred_visibility[0, -1, :], :]
		
		queries_ref = torch.clone(queries)
		queries_ref = queries_ref[:, pred_visibility[0, -1, :], :]


		pred_tracks_reverse, pred_visibility_reverse = self.tracking_network(image_seq_reverse, queries=queries_reverse)
		



		for i in range(pred_visibility_reverse.shape[-1]):
			if not pred_visibility_reverse[0, -1, i]:
				continue
			d_vec = queries_ref[0, i, 1:] - pred_tracks_reverse[0, -1, i, :]
			d2 = d_vec[0]*d_vec[0] + d_vec[1]*d_vec[1]
			#print(d_vec, d2)
			if d2 > 50:
				pred_visibility_reverse[0, -1, i] = False

			#breakpoint()
		
		visible_point_ratio_reverse = pred_visibility_reverse[0, -1, :].long().sum() / len(pred_visibility_reverse[0, -1, :]) 
		
		if visible_point_ratio_reverse < 0.8:
			# Too little points visible, likely to gone out of focus
			return False, []
		
		print(f"LOOP FOUND {frame_idx}, {reference_idx}")

		# Visualization for potential loop closure
		img_ref = torch.clone(image_seq[0, 0])
		c = 0
		for point_2d in pred_tracks[0, 0]:
				point_2d_org_x = min(max(4, int(point_2d[0])), img_ref.shape[2]-5)
				point_2d_org_y = min(max(4, int(point_2d[1])), img_ref.shape[1]-5)
				color = get_color(local_point_ids[c])
				c +=1
				if not pred_visibility[0, 0, c-1]:
					continue


				size = 3

				for i in range(-size, size+1):
					for j in range(-size, size+1):
						img_ref[0, point_2d_org_y+j, point_2d_org_x+i] = color[0]
						img_ref[1, point_2d_org_y+j, point_2d_org_x+i] = color[1]
						img_ref[2, point_2d_org_y+j, point_2d_org_x+i] = color[2]

			
		img_tgt = torch.clone(image_seq[0, -1])
		c = 0
		for point_2d in pred_tracks_reverse[0, 0]:
			point_2d_org_x = min(max(4, int(point_2d[0])), img_tgt.shape[2]-5)
			point_2d_org_y = min(max(4, int(point_2d[1])), img_tgt.shape[1]-5)
			color = get_color(local_point_ids_reverse[c])

			c +=1
			if not pred_visibility_reverse[0, -1, c-1]:
				continue
			size = 3

			for i in range(-size, size+1):
				for j in range(-size, size+1):
					img_tgt[0, point_2d_org_y+j, point_2d_org_x+i] = color[0]
					img_tgt[1, point_2d_org_y+j, point_2d_org_x+i] = color[1]
					img_tgt[2, point_2d_org_y+j, point_2d_org_x+i] = color[2]

		print(f"{frame_idx}, {reference_idx}")
		save_image(img_ref, "ref.png")
		save_image(img_tgt, "tgt.png")

		image_tgt = sample_cur['image'].detach().cpu().numpy()
		depth_tgt = self.depth_estimator(sample_cur['image'], sample_cur['mask']).squeeze().detach().cpu().numpy()
		intrinsics_tgt = sample_cur['intrinsics'].detach().cpu().numpy()
		mask_tgt = sample_cur['mask'].squeeze().detach().cpu().numpy()
		mask_tgt[depth_tgt < 1e-6] = 0
		##
		if frame_idx not in slam_structure.keyframes:
			intial_pose_guess, _ = slam_structure.poses[reference_idx]
			slam_structure.add_keypose(frame_idx, intial_pose_guess, intrinsics_tgt, image_tgt, depth_tgt, mask_tgt)

		H, W = mask_tgt.shape
		tgt_pose_points = slam_structure.get_pose_points(frame_idx)
		tgt_point_ids = []
		for (point_id, point2d) in tgt_pose_points:
			tgt_point_ids.append(point_id)

		# Add correspondences
		for i in range(len(local_point_ids)):
			point_id = local_point_ids[i]
			tracked_point = pred_tracks[0, -1, i, :].detach().cpu().numpy()

			if tracked_point[0] < 0 or tracked_point[1] < 0 or tracked_point[0] >= W or tracked_point[1] >= H:
				continue

			if mask_tgt[int(tracked_point[1]),  int(tracked_point[0])] == 0:
				continue

			if point_id in tgt_point_ids:
				continue

			if pred_visibility[0, -1, i]:
				#breakpoint()
				slam_structure.add_pose_point(frame_idx, point_id, tracked_point)

		#breakpoint()

		return True, [reference_idx, frame_idx]
     

	def close_loops(self, frame_idx, dataset, slam_structure):
		loops = []
		for reference_frame in self.reference_frames:
			if len(loops) > 0:
				continue
			found_loop, loop = self.check_and_close_loop(frame_idx, reference_frame, dataset, slam_structure)
			if found_loop:
				loops.append(loop)
		
		return loops 

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
		#set frames to skip avoid minimum motion
		self.skip_frames = 300
		self.start_idx = start_idx
		self.end_idx = end_idx
		self.downsample_imagedict()
		self.tree, self.matcher = r2d2_init(self.img_dict)
		self.vis_save_path = '../../temp_data/loop_detect/r2d2'
		remake_vis_dir(self.vis_save_path)
  
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
		kps, des = r2d2.update_image(image, 200)
		self.kps_dict[idx] = kps
		if des is not None:
			self.tree.update_tree(idx, des)
		res = {}
		# skip 100 frames to compare the similarity
		for j in self.tree.imageIDs[:-1]:
			if abs(j - idx) < self.skip_frames:
				continue
			# print('Image {} vs Image {}: {}'.format(idx, j, self.matcher.cos_sim(self.tree.transform(idx), self.tree.transform(j))), end='\r')
			if self.matcher.cos_sim(self.tree.transform(idx), self.tree.transform(j)) >= self.T:
				res[j] = self.matcher.cos_sim(self.tree.transform(idx), self.tree.transform(j))
		if res:
			r = max(res.items(), key=lambda x:x[1])[0]
			print (f"Image {idx} is similar to Image {r} with similarity: ",res[r] )
   			# save loop frames to visualize result
			vis_loop_frames(self.img_dict, idx, r, self.kps_dict, save_path=self.vis_save_path)
			return True, [r, idx]
		else:
			return False, []

	def close_loops(self, frame_idx):

		found_loop, loop = self.r2d2_loop_detect(frame_idx)
		if found_loop:
			return loop 

	def find_motion(self, loops):
		loop_mes_list = []
		pose = np.identity(4)
		intrinsics = self.dataset[loops[0][0]]['intrinsics']
		K = np.identity(3)
		K[0,0] = intrinsics[0]
		K[0,1] = intrinsics[1]
		K[0,2] = intrinsics[2]
		K[1,2] = intrinsics[3]
		num_points = 1000
  
		for [ref_id, query_id] in loops:
			ref_img = self.dataset[ref_id]['image'].permute(1, 2, 0).detach().cpu().numpy()*255
			ref_keypoints, ref_des = r2d2.update_image(ref_img, num_points)
			query_img = self.dataset[query_id]['image'].permute(1, 2, 0).detach().cpu().numpy()*255
			query_keypoints, query_des = r2d2.update_image(query_img, num_points)
			# FLANN parameters
			FLANN_INDEX_KDTREE = 1
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
			search_params = dict(checks=50) # or pass empty dictionary
			flann = cv2.FlannBasedMatcher(index_params,search_params)
			ref_des = np.array(ref_des).astype(np.float32)
			query_des = np.array(query_des).astype(np.float32)
			matches = flann.knnMatch(ref_des, query_des, k=2)

			good_matches = []
			for m, n in matches:
				if m.distance < 0.75 * n.distance:
					good_matches.append(m)
			
			keypoints1 = ref_keypoints[:,:2]
			keypoints2 = query_keypoints[:,:2]
			keypoints1 = [cv2.KeyPoint(x, y, 1) for x, y in keypoints1]
			keypoints2 = [cv2.KeyPoint(x, y, 1) for x, y in keypoints2]
			
			ref_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
			query_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

			# Minimal motion between frames
			if np.linalg.norm(query_keypoints2 - ref_keypoints1).mean() < 5:
				print('Minimal motion.')
				R = np.identity(3)
				t[:] = 0
			else:
				E, mask = cv2.findEssentialMat(query_keypoints2, ref_keypoints1, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
				_, R, t, _ = cv2.recoverPose(E, query_keypoints2, ref_keypoints1, cameraMatrix=K)
			pose[:3, :3] = R
			pose[:3, 3] = t.reshape(3,)
   
			loop_mes_list.append([ref_id, query_id, pose])
		return loop_mes_list
