
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
        for idx in self.slam_structure.keyframes[:-(local_ba_size+new_keyframe_counter)]:
            self.localBA.BA.fix_pose(idx, fixed=True)
        for idx in self.slam_structure.keyframes[-(local_ba_size+new_keyframe_counter):]:
            self.localBA.BA.fix_pose(idx, fixed=False)
        self.localBA.BA.fix_pose(self.slam_structure.keyframes[0], fixed=True)
        # print('local BA start................................')
        # print(f'fix: {slam_structure.keyframes[:-(args.local_ba_size+new_keyframe_counter)]}')
        # print(f'unfix: {slam_structure.keyframes[-(args.local_ba_size+new_keyframe_counter):]}')
        self.localBA.run_ba(opt_iters=tracking_ba_iterations)
    
    # def covisible(self):
    #     for idx in self.slam_structure.keyframes:
    #         points = self.slam_structure.pose_point_map[idx]
    #         print(f'covisible for frame {idx}', points)
    #         # print(self.slam_structure.covisible_frames[idx])