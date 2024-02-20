from itertools import chain
import logging
import g2o
from misc.components import Measurement
from optimizers import LocalBA_cg
import numpy as np
logger = logging.getLogger('logfile.txt')

class mapping:
    def __init__(self, slam_structure, localBA, graph, local_ba_size) -> None:
        self.slam_structure = slam_structure
        # self.point_resample_cooldown = point_resample_cooldown
        self.localBA = localBA
        self.local_ba_size = local_ba_size
        self.local_keyframes = []
        self.keyframe_list = []
        self.prevFlag = 0
        self.cutFlag = 0
        self.keyframe_counter = 0
        self.all = True
        self.graph = graph 
        self.localBA_cg = LocalBA_cg()
        
    def add_keyframe(self, keyframe, measurements):
        self.graph.add_keyframe(keyframe)
        self.slam_structure.create_points(keyframe)

        for m in measurements:
            self.graph.add_measurement(keyframe, m.mappoint, m)

        self.local_keyframes.clear()
        self.local_keyframes.append(keyframe)

        self.fill(self.local_keyframes, keyframe)
        # self.refind(self.local_keyframes, self.get_owned_points(keyframe))

        self.bundle_adjust(self.local_keyframes)
        self.points_culling(self.local_keyframes)
    
    def get_owned_points(self, keyframe):
        owned = []
        for m in keyframe.measurements():
            if m.from_mapping():
                owned.append(m.mappoint)
        return owned
    
    def fill(self, keyframes, keyframe):
        covisible = sorted(
            keyframe.covisibility_keyframes().items(), 
            key=lambda _:_[1], reverse=True)

        for kf, n in covisible:
            if n > 0 and kf not in keyframes and self.is_safe(kf):
                keyframes.append(kf)
                if len(keyframes) >= self.local_ba_size:
                    return
    
    def refind(self, keyframes, new_mappoints):    # time consuming
        if len(new_mappoints) == 0:
            return
        for keyframe in keyframes:
            filtered = self.filter_unmatched_points(keyframe, new_mappoints)
            if len(filtered) == 0:
                continue
            for mappoint in filtered:
                mappoint.increase_projection_count()

            measuremets = keyframe.match_mappoints(filtered, Measurement.Source.REFIND)

            for m in measuremets:
                self.graph.add_measurement(keyframe, m.mappoint, m)
                m.mappoint.increase_measurement_count()
                
    def filter_unmatched_points(self, keyframe, mappoints):
        filtered = []
        for i in np.where(keyframe.can_view(mappoints))[0]:
            pt = mappoints[i]
            if (not pt.is_bad() and 
                not self.graph.has_measurement(keyframe, pt)):
                filtered.append(pt)
        return filtered
    
    def bundle_adjust(self, keyframes):
        adjust_keyframes = set()
        for kf in keyframes:
            if not kf.is_fixed():
                adjust_keyframes.add(kf)

        fixed_keyframes = set()
        for kf in adjust_keyframes:
            for ck, n in kf.covisibility_keyframes().items():
                if (n > 0 and ck not in adjust_keyframes 
                    and self.is_safe(ck) and ck < kf):
                    fixed_keyframes.add(ck)

        self.localBA_cg.set_data(adjust_keyframes, fixed_keyframes)
        completed = self.localBA_cg.optimize(10)

        self.localBA_cg.update_poses()
        self.localBA_cg.update_points()
        
        bad_measurements = self.localBA_cg.get_bad_measurements()
        self.remove_measurements(bad_measurements)
        
        bad_ratio = len(bad_measurements)/self.localBA_cg.edge_count

        if bad_ratio < 0.9:
            pass
        else:
            logger.info(f'{self.keyframe_list[-1].idx} high bad ratio, aborting local BA')
            # self.cutFlag += 1
        
        fix_idx = [fixed_keyframe.idx for fixed_keyframe in fixed_keyframes]
        unfix_idx = [adjust_keyframe.idx for adjust_keyframe in adjust_keyframes]
        logger.info(f'Local BA info: bad:{len(bad_measurements)}, bad ratio:{len(bad_measurements)/(self.localBA_cg.edge_count)}, total: {len(self.localBA_cg.optimizer.active_edges())}, Window: {fix_idx}#{unfix_idx}#')
        self.localBA_cg.edge_count = 0
    
    def remove_measurements(self, measurements):
        for m in measurements:
            m.mappoint.increase_outlier_count()
            self.graph.remove_measurement(m)
    
    def points_culling(self, keyframes):    # Remove bad mappoints
        mappoints = set(chain(*[kf.mappoints() for kf in keyframes]))
        for pt in mappoints:
            if pt.is_bad():
                self.graph.remove_mappoint(pt)
    
    def is_safe(self, keyframe):
        return True
     # ***********************************************************************************
     
    def local_BA(self, tracking_ba_iterations, new_keyframe_counter):
         # If there are new keyframes, run local BA
        ba_size = 3*self.local_ba_size
        self.slam_structure.key_frames = dict(sorted(self.slam_structure.key_frames.items()))
        self.keyframe_list = list(self.slam_structure.key_frames.values())
        ###################################################################################
        # test new local BA optimizer for each new keyframe 
        fix_idx = []
        unfix_idx = []
        fix2_idx = []
        for keyframe in self.keyframe_list[:-(self.local_ba_size+new_keyframe_counter)]:
            # self.localBA.BA.fix_pose(keyframe.idx, fixed=True)
            fix_idx.append(keyframe.idx)
        for keyframe in self.keyframe_list[-(self.local_ba_size+new_keyframe_counter):]:
            # self.localBA.BA.fix_pose(keyframe.idx, fixed=False)
            unfix_idx.append(keyframe.idx)
            
        # for the local BA after initial full BA    
        if self.keyframe_list[1].idx in unfix_idx:
            if self.keyframe_list[0].idx in unfix_idx:
                fix_idx.append(self.keyframe_list[0].idx)
                fix_idx.append(self.keyframe_list[1].idx)
                unfix_idx.remove(self.keyframe_list[0].idx)
                unfix_idx.remove(self.keyframe_list[1].idx)
            else:
                fix_idx.append(self.keyframe_list[1].idx)
                unfix_idx.remove(self.keyframe_list[1].idx)
        
        if len(unfix_idx) + len(fix_idx) > ba_size:
            res = len(unfix_idx) + len(fix_idx) - ba_size
            fix_idx = fix_idx[res:]
        
        self.localBA.set_data(fix_idx, unfix_idx)
        # Remove bad measurements
        self.localBA.run_ba(opt_iters=tracking_ba_iterations)
        bad_measurements = self.localBA.get_bad_measurements()
        self.localBA.remove_bad_measurements()
        
        count = 0
        for edge in self.localBA.edge_info:
            if edge[0] == self.keyframe_list[-1].idx:
                count += 1
        
        bad_ratio = len(bad_measurements)/count
        if bad_ratio < 0.9:
            self.localBA.extract_ba_data()
        else:
            logger.info(f'{self.keyframe_list[-1].idx} high bad ratio, aborting local BA')
            
        logger.info(f'Local BA info: bad:{len(bad_measurements)}, bad ratio:{len(bad_measurements)/(count+0.0001)}, total: {len(self.localBA.BA.active_edges())}, Window: {fix_idx}#{unfix_idx}#{fix2_idx}')

        # ###################################################################################
        # # test covisibility graph
        # # fix_idx = []
        # # unfix_idx = []
        # fix2_idx = []
        # # for keyframe in self.keyframe_list[:-(self.local_ba_size+new_keyframe_counter)]:
        # #     # self.localBA.BA.fix_pose(keyframe.idx, fixed=True)
        # #     fix_idx.append(keyframe.idx)
        # # for keyframe in self.keyframe_list[-(self.local_ba_size+new_keyframe_counter):]:
        # #     # self.localBA.BA.fix_pose(keyframe.idx, fixed=False)
        # #     unfix_idx.append(keyframe.idx)
            
        # # # for the local BA after initial full BA    
        # # if self.keyframe_list[1].idx in unfix_idx:
        # #     if self.keyframe_list[0].idx in unfix_idx:
        # #         fix_idx.append(self.keyframe_list[0].idx)
        # #         fix_idx.append(self.keyframe_list[1].idx)
        # #         unfix_idx.remove(self.keyframe_list[0].idx)
        # #         unfix_idx.remove(self.keyframe_list[1].idx)
        # #     else:
        # #         fix_idx.append(self.keyframe_list[1].idx)
        # #         unfix_idx.remove(self.keyframe_list[1].idx)
        
        # # self.local_keyframes.clear()
        # self.local_keyframes.append(keyframe)

        # self.fill(self.local_keyframes, keyframe)
        # fix_idx=[self.keyframe_list[0].idx, self.keyframe_list[1].idx]
        # unfix_idx = [keyframe.idx for keyframe in self.local_keyframes]
        # print('fix_idx:', fix_idx, 'unfix_idx:', unfix_idx)
        # self.localBA.set_data(fix_idx, unfix_idx)
        # # Remove bad measurements
        # self.localBA.run_ba(opt_iters=10)
        # bad_measurements = self.localBA.get_bad_measurements()
        # self.localBA.remove_bad_measurements()
        
        # count = 0
        # for edge in self.localBA.edge_info:
        #     if edge[0] == self.keyframe_list[-1].idx:
        #         count += 1
        
        # bad_ratio = len(bad_measurements)/count
        # if bad_ratio < 0.5:
        #     try:
        #         self.localBA.extract_ba_data()
        #     except:
        #         pass
        # else:
        #     logger.info(f'{self.keyframe_list[-1].idx} high bad ratio, aborting local BA.')

           
        # logger.info(f'Local BA info: bad:{len(bad_measurements)}, bad ratio:{len(bad_measurements)/count}, total: {len(self.localBA.BA.active_edges())}, Window: {fix_idx}#{unfix_idx}#{fix2_idx}')

        # return bad_measurements
        
    
    ###################################################################################
        # test new local BA optimizer for each new keyframe 
    def full_BA(self):
        fix_idx = []
        unfix_idx = []
        self.keyframe_list = list(self.slam_structure.key_frames.values())[:2]
        for keyframe in self.keyframe_list:
            idx = keyframe.idx
            unfix_idx.append(idx)
            
        if self.keyframe_list[0].idx not in fix_idx:
            fix_idx.append(self.keyframe_list[0].idx)
            unfix_idx.remove(self.keyframe_list[0].idx)
        print('Full BA start......')
        self.localBA.set_data(fix_idx, unfix_idx)
        # Remove bad measurements
        self.localBA.run_ba(opt_iters=10)
        bad_measurements = self.localBA.get_bad_measurements()
        self.localBA.remove_bad_measurements()
        
        self.localBA.extract_ba_data()

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