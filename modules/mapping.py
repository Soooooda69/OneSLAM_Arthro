import logging
import g2o
logger = logging.getLogger('logfile.txt')

class mapping:
    def __init__(self, slam_structure, localBA, local_ba_size) -> None:
        self.slam_structure = slam_structure
        # self.point_resample_cooldown = point_resample_cooldown
        self.localBA = localBA
        self.local_ba_size = local_ba_size
        self.local_keyframes = []
        
    def local_BA(self, tracking_ba_iterations, new_keyframe_counter):
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
        # ###################################################################################
        # fix_idx = []
        # unfix_idx = []
        # fix2_idx = []
        # for keyframe in keyframe_list[:-(local_ba_size+new_keyframe_counter)]:
        #     self.localBA.BA.fix_pose(keyframe.idx, fixed=True)
        #     fix_idx.append(keyframe.idx)
        # for keyframe in keyframe_list[-(local_ba_size+new_keyframe_counter):]:
        #     self.localBA.BA.fix_pose(keyframe.idx, fixed=False)
        #     unfix_idx.append(keyframe.idx)
        
        # self.localBA.BA.fix_pose(keyframe_list[0].idx, fixed=True)
            
        # # Remove bad measurements
        # self.localBA.run_ba(opt_iters=10)
        # bad_measurements = self.localBA.get_bad_measurements()
        # # self.localBA.run_ba(opt_iters=10)
        # # bad_measurements = self.localBA.get_bad_measurements()
        
        # count = 0
        # for edge in self.localBA.edge_info:
        #     if edge[0] == keyframe_list[-1].idx:
        #         count += 1
        
        # bad_ratio = len(bad_measurements)/count
        # if bad_ratio < 0.5:
        #     self.localBA.extract_ba_data()
        # else:
        #     logger.info(f'{keyframe_list[-1].idx} high bad ratio, aborting local BA.')

           
        # logger.info(f'Local BA info: bad:{len(bad_measurements)}, bad ratio:{len(bad_measurements)/count}, total: {len(self.localBA.BA.active_edges())}, Window: {fix_idx}#{unfix_idx}#{fix2_idx}')

        # return bad_measurements
        ###################################################################################
        # test new local BA optimizer for each new keyframe 
        fix_idx = []
        unfix_idx = []
        fix2_idx = []
        for keyframe in keyframe_list[:-(self.local_ba_size+new_keyframe_counter)]:
            # self.localBA.BA.fix_pose(keyframe.idx, fixed=True)
            fix_idx.append(keyframe.idx)
        for keyframe in keyframe_list[-(self.local_ba_size+new_keyframe_counter):]:
            # self.localBA.BA.fix_pose(keyframe.idx, fixed=False)
            unfix_idx.append(keyframe.idx)
            
        # for the local BA after initial full BA    
        if keyframe_list[1].idx in unfix_idx:
            if keyframe_list[0].idx in unfix_idx:
                fix_idx.append(keyframe_list[0].idx)
                fix_idx.append(keyframe_list[1].idx)
                unfix_idx.remove(keyframe_list[0].idx)
                unfix_idx.remove(keyframe_list[1].idx)
            else:
                fix_idx.append(keyframe_list[1].idx)
                unfix_idx.remove(keyframe_list[1].idx)
        
        self.localBA.set_data(fix_idx, unfix_idx)
        # Remove bad measurements
        self.localBA.run_ba(opt_iters=10)
        bad_measurements = self.localBA.get_bad_measurements()
        self.localBA.remove_bad_measurements()
        
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
        # ###################################################################################
        # # test covisibility graph
        # # fix_idx = []
        # # unfix_idx = []
        # fix2_idx = []
        # # for keyframe in keyframe_list[:-(self.local_ba_size+new_keyframe_counter)]:
        # #     # self.localBA.BA.fix_pose(keyframe.idx, fixed=True)
        # #     fix_idx.append(keyframe.idx)
        # # for keyframe in keyframe_list[-(self.local_ba_size+new_keyframe_counter):]:
        # #     # self.localBA.BA.fix_pose(keyframe.idx, fixed=False)
        # #     unfix_idx.append(keyframe.idx)
            
        # # # for the local BA after initial full BA    
        # # if keyframe_list[1].idx in unfix_idx:
        # #     if keyframe_list[0].idx in unfix_idx:
        # #         fix_idx.append(keyframe_list[0].idx)
        # #         fix_idx.append(keyframe_list[1].idx)
        # #         unfix_idx.remove(keyframe_list[0].idx)
        # #         unfix_idx.remove(keyframe_list[1].idx)
        # #     else:
        # #         fix_idx.append(keyframe_list[1].idx)
        # #         unfix_idx.remove(keyframe_list[1].idx)
        
        # # self.local_keyframes.clear()
        # self.local_keyframes.append(keyframe)

        # self.fill(self.local_keyframes, keyframe)
        # fix_idx=[keyframe_list[0].idx, keyframe_list[1].idx]
        # unfix_idx = [keyframe.idx for keyframe in self.local_keyframes]
        # print('fix_idx:', fix_idx, 'unfix_idx:', unfix_idx)
        # self.localBA.set_data(fix_idx, unfix_idx)
        # # Remove bad measurements
        # self.localBA.run_ba(opt_iters=10)
        # bad_measurements = self.localBA.get_bad_measurements()
        # self.localBA.remove_bad_measurements()
        
        # count = 0
        # for edge in self.localBA.edge_info:
        #     if edge[0] == keyframe_list[-1].idx:
        #         count += 1
        
        # bad_ratio = len(bad_measurements)/count
        # if bad_ratio < 0.5:
        #     try:
        #         self.localBA.extract_ba_data()
        #     except:
        #         pass
        # else:
        #     logger.info(f'{keyframe_list[-1].idx} high bad ratio, aborting local BA.')

           
        # logger.info(f'Local BA info: bad:{len(bad_measurements)}, bad ratio:{len(bad_measurements)/count}, total: {len(self.localBA.BA.active_edges())}, Window: {fix_idx}#{unfix_idx}#{fix2_idx}')

        # return bad_measurements
        
    # def full_BA(self):
    #     keyframe_list = list(self.slam_structure.key_frames.values())
    #     for keyframe in self.slam_structure.key_frames.values():
    #         idx = keyframe.idx
    #         self.localBA.BA.fix_pose(idx, fixed=False)

    #     self.localBA.BA.fix_pose(keyframe_list[0].idx, fixed=True)
    #     print('Full BA start......')
        
    #     # Remove bad measurements
    #     self.localBA.run_ba(opt_iters=10)
    #     bad_measurements = self.localBA.get_bad_measurements()
                    
    #     frame_edges = {}
    #     for edge in self.localBA.edge_info:
    #         frame_idx = edge[0]
    #         if frame_idx not in frame_edges:
    #             frame_edges[frame_idx] = 1
    #         else:
    #             frame_edges[frame_idx] += 1
        
    #     for frame_idx, num_edges in frame_edges.items():
    #         print(f"Frame {frame_idx}: {num_edges} edges")
        
    #     logger.info(f'Initial Full BA info: bad:{len(bad_measurements)}, bad ratio:{len(bad_measurements)/len(self.localBA.BA.active_edges())}, total: {len(self.localBA.BA.active_edges())}')

    #     return bad_measurements
    
    ###################################################################################
        # test new local BA optimizer for each new keyframe 
    def full_BA(self):
        fix_idx = []
        unfix_idx = []
        keyframe_list = list(self.slam_structure.key_frames.values())
        for keyframe in self.slam_structure.key_frames.values():
            idx = keyframe.idx
            unfix_idx.append(idx)
            # self.localBA.BA.fix_pose(idx, fixed=False)
        if keyframe_list[0].idx not in fix_idx:
            fix_idx.append(keyframe_list[0].idx)
            unfix_idx.remove(keyframe_list[0].idx)

        print('Full BA start......')
        self.localBA.set_data(fix_idx, unfix_idx)
        # Remove bad measurements
        self.localBA.run_ba(opt_iters=10)
        bad_measurements = self.localBA.get_bad_measurements()
        self.localBA.remove_bad_measurements()
        
        self.localBA.extract_ba_data()

        # frame_edges = {}
        # for edge in self.localBA.edge_info:
        #     frame_idx = edge[0]
        #     if frame_idx not in frame_edges:
        #         frame_edges[frame_idx] = 1
        #     else:
        #         frame_edges[frame_idx] += 1
        
        # for frame_idx, num_edges in frame_edges.items():
        #     print(f"Frame {frame_idx}: {num_edges} edges")
        
        logger.info(f'Initial Full BA info: bad:{len(bad_measurements)}, bad ratio:{len(bad_measurements)/len(self.localBA.BA.active_edges())}, total: {len(self.localBA.BA.active_edges())}')

        return bad_measurements
    
    # def fill(self, keyframes, keyframe):
    #     covisible = sorted(
    #         keyframe.covisibility_keyframes().items(), 
    #         key=lambda _:_[1], reverse=True)
    #     print('cccccccccccccccccccccccc:', covisible)
    #     for kf, n in covisible:
    #         if n > 0 and kf not in keyframes and self.is_safe(kf):
    #             print('fill:', kf.idx)
    #             keyframes.append(kf)
    #             if len(keyframes) >= self.local_ba_size:
    #                 return