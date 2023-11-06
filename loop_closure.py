import torch
import numpy as np

class LoopClosure:
	def __init__(self, loop_checkers, point_tracker):
		self.loop_checkers = loop_checkers
		self.point_tracker = point_tracker
		self.all_loops = []

	# Adding new reference frames
	def add_reference_frame(self, frame_idx):
		for loop_checker in self.loop_checkers:
			loop_checker.add_reference_frame(frame_idx)


	def close_loops(self, frame_idx, dataset, slam_structure):
		ref_frames = []

		# Loop searching
		for loop_checker in self.loop_checkers:
			ref_frames += loop_checker.search_loops(frame_idx, dataset, slam_structure) # [ref_frame]

		if len(ref_frames) == 0:
			return []
		
		# For performance sanity: Only use loop with oldest frame_idx
		ref_frames = sorted(ref_frames)
		loops = [(ref_frames[0], frame_idx)]

		# Point tracking
		for loop in loops:
			print(f"Tracking points for loop {loop}.")
			section = [loop[0], loop[0], loop[0], loop[0], loop[1], loop[1], loop[1], loop[1]]
			section_keyframes = [loop[0], loop[1]]
			self.point_tracker.process_section(section, section_keyframes, dataset, slam_structure, sample_new_points=False)
		
		return loops 

		