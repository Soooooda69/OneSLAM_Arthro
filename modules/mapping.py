
class mapping:
    def __init__(self, slam_structure, localBA, point_resample_cooldown) -> None:
        self.slam_structure = slam_structure
        self.point_resample_cooldown = point_resample_cooldown
        self.localBA = localBA
        
    def remove_correspondences(self, current_section):
        # Remove all existing correspondences (likely to be faulty),
        # except for frist frame
        for idx in current_section[1:]:
            # print(f'removing correspondences for frame {idx}', current_section)
            # assert idx not in self.slam_structure.keyframes
            self.slam_structure.pose_point_map[idx] = []

    def local_BA(self, local_ba_size, tracking_ba_iterations, new_keyframe_counter):
         # If there are new keyframes, run local BA
        # for idx in self.slam_structure.keyframes[:-(local_ba_size+new_keyframe_counter)]:
        self.slam_structure.key_frames = dict(sorted(self.slam_structure.key_frames.items()))
        keyframe_list = list(self.slam_structure.key_frames.values())
        for keyframe in keyframe_list[:-(local_ba_size+new_keyframe_counter)]:
            idx = keyframe.idx
            self.localBA.BA.fix_pose(idx, fixed=True)
            # for (point_id,_) in self.slam_structure.pose_point_map[idx]:
            #     self.localBA.BA.fix_point(point_id, fixed=True)
        # for idx in self.slam_structure.keyframes[-(local_ba_size+new_keyframe_counter):]:
        for keyframe in keyframe_list[-(local_ba_size+new_keyframe_counter):]:
            idx = keyframe.idx
            self.localBA.BA.fix_pose(idx, fixed=False)
            # for (point_id,_) in self.slam_structure.pose_point_map[idx]:
            #     self.localBA.BA.fix_point(point_id, fixed=False)
        self.localBA.BA.fix_pose(keyframe_list[0].idx, fixed=True)
        print('local BA start......')
        
        # Remove bad measurements
        self.localBA.run_ba(opt_iters=10)
        bad_measurements = self.localBA.get_bad_measurements()
        self.localBA.run_ba(opt_iters=10)
        # bad_measurements = self.localBA.get_bad_measurements()
        return bad_measurements