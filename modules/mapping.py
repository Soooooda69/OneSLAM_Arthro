import logging
import g2o
logger = logging.getLogger('logfile.txt')

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
        
        
        # If there are new keyframes, run local BA for each new keyframe
        for i in range(new_keyframe_counter):
            fix_idx = []
            unfix_idx = []
            fix2_idx = []
            window_end = new_keyframe_counter-i-1
            window_start = window_end+local_ba_size
            if window_end == 0:
                # unfix all within window except the 1st keyframe
                for keyframe in keyframe_list[:-(window_start)]:
                    idx = keyframe.idx
                    self.localBA.BA.fix_pose(idx, fixed=True)
                    # self.localBA.BA.add_pose(idx, keyframe.pose, keyframe.intrinsics, fixed=True)
                    fix_idx.append(idx)
                    
                for keyframe in keyframe_list[-(window_start):]:
                    idx = keyframe.idx
                    self.localBA.BA.fix_pose(idx, fixed=False)
                    # self.localBA.BA.add_pose(idx, keyframe.pose, keyframe.intrinsics, fixed=False)
                    unfix_idx.append(idx)
                
            else:
                for keyframe in keyframe_list[:-(window_start)]:
                    idx = keyframe.idx
                    self.localBA.BA.fix_pose(idx, fixed=True)
                    # self.localBA.BA.add_pose(idx, keyframe.pose, keyframe.intrinsics, fixed=True)
                    fix_idx.append(idx)

                    
                for keyframe in keyframe_list[-(window_start):-(window_end)]:
                    idx = keyframe.idx
                    self.localBA.BA.fix_pose(idx, fixed=False)
                    # self.localBA.BA.add_pose(idx, keyframe.pose, keyframe.intrinsics, fixed=False)
                    unfix_idx.append(idx)
                
                for keyframe in keyframe_list[-(window_end):]:
                    idx = keyframe.idx
                    self.localBA.BA.fix_pose(idx, fixed=True)
                    # self.localBA.BA.add_pose(idx, keyframe.pose, keyframe.intrinsics, fixed=True)
                    fix2_idx.append(idx)

            self.localBA.BA.fix_pose(keyframe_list[0].idx, fixed=True)
            # self.localBA.BA.add_pose(keyframe_list[0].idx, keyframe_list[0].pose, keyframe_list[0].intrinsics, fixed=True)
            if 1 not in fix_idx:
                fix_idx.append(1)
                unfix_idx.remove(1)
            # print(f'Local BA start......{fix_idx}#{unfix_idx}#{fix2_idx}')
            # logger.info(f'Local BA :{fix_idx}#{unfix_idx}#{fix2_idx}')
            
            # Remove bad measurements
            self.localBA.run_ba(opt_iters=10)
            bad_measurements = self.localBA.get_bad_measurements()
            # self.localBA.run_ba(opt_iters=10)
            # bad_measurements = self.localBA.get_bad_measurements()
            
            count = 0
            for edge in self.localBA.BA.edge_info:
                if edge[0] == keyframe_list[-1].idx:
                    count += 1
                
            logger.info(f'Local BA info: bad:{len(bad_measurements)}, bad ratio:{len(bad_measurements)/count}, total: {len(self.localBA.BA.active_edges())}, Window: {fix_idx}#{unfix_idx}#{fix2_idx}')

        return bad_measurements
    
    def full_BA(self):

        # keyframe_list = list(self.slam_structure.key_frames.values())
        for keyframe in self.slam_structure.key_frames.values():
            idx = keyframe.idx
            self.localBA.BA.fix_pose(idx, fixed=False)

        self.localBA.BA.fix_pose(list(self.slam_structure.key_frames.keys())[0], fixed=True)
        print('Full BA start......')
        
        # Remove bad measurements
        self.localBA.run_ba(opt_iters=10)
        bad_measurements = self.localBA.get_bad_measurements()
        # self.localBA.run_ba(opt_iters=10)
        # bad_measurements = self.localBA.get_bad_measurements()
        return bad_measurements