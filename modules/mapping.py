import logging
import g2o
logger = logging.getLogger('logfile.txt')

class mapping:
    def __init__(self, slam_structure, localBA, point_resample_cooldown) -> None:
        self.slam_structure = slam_structure
        self.point_resample_cooldown = point_resample_cooldown
        self.localBA = localBA
        

    def local_BA(self, local_ba_size, tracking_ba_iterations, new_keyframe_counter):
         # If there are new keyframes, run local BA
        self.slam_structure.key_frames = dict(sorted(self.slam_structure.key_frames.items()))
        keyframe_list = list(self.slam_structure.key_frames.values())
        
        # # If there are new keyframes, run local BA for each new keyframe
        # for i in range(new_keyframe_counter):
        #     fix_idx = []
        #     unfix_idx = []
        #     fix2_idx = []
        #     window_end = new_keyframe_counter-i-1
        #     window_start = window_end+local_ba_size
        #     if window_end == 0:
        #         # unfix all within window except the 1st keyframe
        #         for keyframe in keyframe_list[:-(window_start)]:
        #             idx = keyframe.idx
        #             self.localBA.BA.fix_pose(idx, fixed=True)
        #             fix_idx.append(idx)
                    
        #         for keyframe in keyframe_list[-(window_start):]:
        #             idx = keyframe.idx
        #             self.localBA.BA.fix_pose(idx, fixed=False)
        #             unfix_idx.append(idx)
                
        #     else:
        #         for keyframe in keyframe_list[:-(window_start)]:
        #             idx = keyframe.idx
        #             self.localBA.BA.fix_pose(idx, fixed=True)
        #             fix_idx.append(idx)

        #         for keyframe in keyframe_list[-(window_start):-(window_end)]:
        #             idx = keyframe.idx
        #             self.localBA.BA.fix_pose(idx, fixed=False)
        #             unfix_idx.append(idx)
                
        #         for keyframe in keyframe_list[-(window_end):]:
        #             idx = keyframe.idx
        #             self.localBA.BA.fix_pose(idx, fixed=True)
        #             fix2_idx.append(idx)
        
        #     if keyframe_list[0].idx not in fix_idx:
        #         fix_idx.append(keyframe_list[0].idx)
        #         unfix_idx.remove(keyframe_list[0].idx)
            
        #     self.localBA.BA.fix_pose(keyframe_list[0].idx, fixed=True)
            
        #     # Remove bad measurements
        #     self.localBA.run_ba(opt_iters=10)
        #     bad_measurements = self.localBA.get_bad_measurements()
        #     # self.localBA.run_ba(opt_iters=10)
        #     # bad_measurements = self.localBA.get_bad_measurements()
            
        #     count = 0
        #     for edge in self.localBA.edge_info:
        #         if edge[0] == keyframe_list[-1].idx:
        #             count += 1
            
        #     bad_ratio = len(bad_measurements)/count
        #     if bad_ratio < 0.5:
        #         self.localBA.extract_ba_data()
        #     else:
        #         logger.info(f'{keyframe_list[-1].idx} high bad ratio, aborting local BA.')
        #         # break
            
        #     logger.info(f'Local BA info: bad:{len(bad_measurements)}, bad ratio:{bad_ratio}, total: {len(self.localBA.BA.active_edges())}, Window: {fix_idx}#{unfix_idx}#{fix2_idx}')

        # return bad_measurements
    
        fix_idx = []
        unfix_idx = []
        fix2_idx = []
        for keyframe in keyframe_list[:-(local_ba_size+new_keyframe_counter)]:
            self.localBA.BA.fix_pose(keyframe.idx, fixed=True)
            fix_idx.append(keyframe.idx)
        for keyframe in keyframe_list[-(local_ba_size+new_keyframe_counter):]:
            self.localBA.BA.fix_pose(keyframe.idx, fixed=False)
            unfix_idx.append(keyframe.idx)
        
        self.localBA.BA.fix_pose(keyframe_list[0].idx, fixed=True)
            
        # Remove bad measurements
        self.localBA.run_ba(opt_iters=10)
        bad_measurements = self.localBA.get_bad_measurements()
        # self.localBA.run_ba(opt_iters=10)
        # bad_measurements = self.localBA.get_bad_measurements()
        
        count = 0
        for edge in self.localBA.edge_info:
            if edge[0] == keyframe_list[-1].idx:
                count += 1
        
        bad_ratio = len(bad_measurements)/count
        if bad_ratio < 0.5:
            self.localBA.extract_ba_data()
        else:
            logger.info(f'{keyframe_list[-1].idx} high bad ratio, aborting local BA.')

           
        logger.info(f'Local BA info: bad:{len(bad_measurements)}, bad ratio:{len(bad_measurements)/count}, total: {len(self.localBA.BA.active_edges())}, Window: {fix_idx}#{unfix_idx}#{fix2_idx}')

        return bad_measurements
    
    def full_BA(self):
        keyframe_list = list(self.slam_structure.key_frames.values())
        for keyframe in self.slam_structure.key_frames.values():
            idx = keyframe.idx
            self.localBA.BA.fix_pose(idx, fixed=False)

        self.localBA.BA.fix_pose(keyframe_list[0].idx, fixed=True)
        print('Full BA start......')
        
        # Remove bad measurements
        self.localBA.run_ba(opt_iters=10)
        bad_measurements = self.localBA.get_bad_measurements()
        # self.localBA.run_ba(opt_iters=10)
        # bad_measurements = self.localBA.get_bad_measurements()
        # for keyframe in self.slam_structure.key_frames.values():
        #     for mappoint_idx in keyframe.feature.keypoints_ids:
        #         self.localBA.BA.fix_point(mappoint_idx, fixed=True)
                    
        frame_edges = {}
        for edge in self.localBA.edge_info:
            frame_idx = edge[0]
            if frame_idx not in frame_edges:
                frame_edges[frame_idx] = 1
            else:
                frame_edges[frame_idx] += 1
        
        for frame_idx, num_edges in frame_edges.items():
            print(f"Frame {frame_idx}: {num_edges} edges")
        
        logger.info(f'Initial Full BA info: bad:{len(bad_measurements)}, bad ratio:{len(bad_measurements)/len(self.localBA.BA.active_edges())}, total: {len(self.localBA.BA.active_edges())}')

        return bad_measurements