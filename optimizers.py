import numpy as np
import g2o
from g2o.contrib import SmoothEstimatePropagator
import logging
from misc.components import Frame, KeyFrame, MapPoint
logger = logging.getLogger(__name__)

class BundleAdjuster(g2o.SparseOptimizer):
    def __init__(self, use_sparse_solver=False):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        if use_sparse_solver:
            solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        # Store edges edge_info[edge_id]: [frame_idx, point_id, point_2d]
        self.edge_info = []
        
    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_pose(self, pose_id, pose, intrinsics, fixed=False):
        fx, fy, cx, cy = intrinsics

        sbacam = g2o.SBACam(pose.orientation(), pose.position())
        sbacam.set_cam(fx, fy, cx, cy, 0)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)   # internal id
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)
    
    # def add_pose_edge(self, pose_indicies, measurement, 
    #         information=np.identity(6),
    #         robust_kernel=None):

    #     edge = g2o.EdgeSE3()
    #     for i, v in enumerate(pose_indicies):
    #         if isinstance(v, int):
    #             v = self.vertex(v*2)
    #         edge.set_vertex(i, v)

    #     edge.set_measurement(measurement)  # relative pose
    #     edge.set_information(information)
    #     if robust_kernel is not None:
    #         edge.set_robust_kernel(robust_kernel)
        
    #     super().add_edge(edge)
    
    

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    # def add_edge(self, point_id, pose_id, 
    #         measurement,
    #         information=np.identity(2),
    #         robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI

    #     edge = g2o.EdgeProjectP2MC()
    #     edge.set_vertex(0, self.vertex(point_id * 2 + 1))
    #     edge.set_vertex(1, self.vertex(pose_id * 2))
    #     edge.set_measurement(measurement)   # projection
    #     edge.set_information(information)

    #     if robust_kernel is not None:
    #         edge.set_robust_kernel(robust_kernel)
    #     super().add_edge(edge)

    #     return edge
    
    def add_edge(self, id, point_id, pose_id, 
            measurement,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI

        edge = g2o.EdgeProjectP2MC()
        edge.set_id(id)
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)

        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

        return edge
    
    def update_edge(self, edge, 
            measurement,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI
        
        edge.set_measurement(measurement)
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)

    def remove_edge(self, edge):
        super().remove_edge(edge)
    
    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()
    
    def set_pose(self, pose_id, pose, intrinsics):
        fx, fy, cx, cy = intrinsics
        pose = g2o.Isometry3d(pose)
        sbacam = g2o.SBACam(pose.orientation(), pose.position())
        sbacam.set_cam(fx, fy, cx, cy, 0)
        self.vertex(pose_id * 2).set_estimate(sbacam)

    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate()
    
    def set_point(self, point_id, point):
        self.vertex(point_id * 2 + 1).set_estimate(point)
        
    def fix_pose(self, pose_id, fixed=True):
        self.vertex(pose_id * 2).set_fixed(fixed)
    
    def fix_point(self, point_id, fixed=True):
        self.vertex(point_id * 2 + 1).set_fixed(fixed)


class LocalBA:
    def __init__(self, slam_structure, BA_sparse_solver, BA_opt_iters, BA_verbose):
        
        # Bundle adjuster + associated settings
        self.BA_sparse_solver = BA_sparse_solver
        self.BA_verbose = BA_verbose
        self.BA_opt_iters = BA_opt_iters
        self.BA = BundleAdjuster(use_sparse_solver=self.BA_sparse_solver)
        self.BA.set_verbose(self.BA_verbose)
        self.slam_sturcture = slam_structure
        # threshold for confidence interval of 95%
        self.huber_threshold = 5.991
        
    def set_frame_data(self, frame_idx, fixed):
        # pose, intrinsics = self.slam_sturcture.poses[frame_idx]
        frame = self.slam_sturcture.all_frames[frame_idx]
        pose = frame.pose
        intrinsics = frame.intrinsic # [fx, fy, cx, cy]
        self.BA.add_pose(frame_idx, g2o.Isometry3d(pose), intrinsics, fixed=fixed)
        
        # Add existing correspondences to BA
        for point_id, (point_2d, des) in frame.feature.keypoints_info.items():
        # for (point_id, point_2d) in self.slam_sturcture.pose_point_map[frame_idx]:
            edge_id = len(self.BA.edge_info)
            edge = self.BA.add_edge(edge_id, point_id, frame_idx, point_2d)
            # print('edge_id:', edge.id())
            # edge = self.BA.add_edge(point_id, frame_idx, point_2d)
            # logger.info(f'frame id: {frame_idx}, add edge:{edge.id()}, point_id:{point_id}.')
            self.BA.edge_info.append([frame_idx, point_id, point_2d])
            if frame_idx not in self.slam_sturcture.pose_point_edges.keys():
                self.slam_sturcture.pose_point_edges[frame_idx] = []
            self.slam_sturcture.pose_point_edges[frame_idx].append(edge)
            
    def get_bad_measurements(self):
        bad_measurements = []
        for edge in self.BA.active_edges():
            if edge.chi2() > self.huber_threshold:
                #remove bad edges
                # self.BA.remove_edge(edge)
                edge.set_level(1)
                bad_measurements.append(self.BA.edge_info[edge.id()])
        # logger.info(f'bad:{len(bad_measurements)}, total: {len(self.BA.active_edges())}, bad ratio:{len(bad_measurements)/len(self.BA.active_edges())}')
        # logger.info(f'bad ratio:{len(bad_measurements)/len(self.BA.active_edges())}')
        # logger.info(f'bad:{bad_measurements}')
        # self.slam_sturcture.remove_measurement(bad_measurements)
        return bad_measurements
    
    def run_ba(self, opt_iters=None):
        if opt_iters is not None:
            self.BA.optimize(opt_iters)
        else:
            self.BA.optimize(self.BA_opt_iters)
        self.valid_frames = set()
        self.extract_ba_data()
    
    # def extract_ba_data(self):
    #     for keyframe in self.slam_sturcture.keyframes:
    #         _, intrinsics = self.slam_sturcture.poses[keyframe]
    #         self.slam_sturcture.poses[keyframe] = (self.BA.get_pose(keyframe).matrix(), intrinsics)      
    #         self.valid_frames.add(keyframe)

    #     for point_id in self.slam_sturcture.points.keys():
    #         _, point_color = self.slam_sturcture.points[point_id]
    #         self.slam_sturcture.points[point_id] = (self.BA.get_point(point_id), point_color)
    #         self.slam_sturcture.valid_points.add(point_id)
    
    def extract_ba_data(self):
        for keyframe in self.slam_sturcture.key_frames.values():
            keyframe.update_pose(self.BA.get_pose(keyframe.idx).matrix())      
            self.valid_frames.add(keyframe)

        # for point_id in self.slam_sturcture.points.keys():
        #     _, point_color = self.slam_sturcture.points[point_id]
        #     self.slam_sturcture.points[point_id] = (self.BA.get_point(point_id), point_color)
        #     self.slam_sturcture.valid_points.add(point_id)
            
        for mappoint in self.slam_sturcture.map_points.values():
            mappoint.update_position(self.BA.get_point(mappoint.id))
            self.slam_sturcture.valid_points.add(mappoint.id)
            
    # def update_ba_data(self):
    #     for point_id in self.slam_sturcture.points.keys():
    #         point, point_color = self.slam_sturcture.points[point_id]
    #         self.BA.set_point(point_id, point)
            
    #     for keyframe in self.slam_sturcture.keyframes:
    #         pose, intrinsics = self.slam_sturcture.poses[keyframe]
    #         self.BA.set_pose(keyframe, pose, intrinsics)
            
    def update_ba_data(self):
        for mappoint in self.slam_sturcture.map_points.values():
            point, point_color = mappoint.position, mappoint.color
            self.BA.set_point(mappoint.id, point)
            
        for keyframe in self.slam_sturcture.key_frames.values():
            pose, intrinsics = keyframe.pose, keyframe.intrinsic
            self.BA.set_pose(keyframe.idx, pose, intrinsics)
            
class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self, slam_structure):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        self.slam_structure = slam_structure
        self.visitedIDs = set()
        
    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, 
            measurement=None, 
            information=np.identity(6),
            robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        if measurement is None:
            measurement = (
                edge.vertex(0).estimate().inverse() * 
                edge.vertex(1).estimate())
        edge.set_measurement(measurement)
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def set_data(self, loops):
        super().clear()
        # keyframe_ids = self.slam_structure.keyframes
        keyframes = self.slam_structure.key_frames
        anchor=None
        for [kf, *_] in loops:
            if anchor is None or kf < anchor:
                anchor = kf
                
        for i, keyframe in enumerate(keyframes.values()):
            # pose = g2o.Isometry3d(self.slam_structure.poses[kf][0])
            pose = keyframe.pose
            fixed = i == 0
            if anchor is not None:
                fixed = keyframe.idx <= anchor
            self.add_vertex(keyframe.idx, pose, fixed=fixed)
            # logger.info(f'add vertex:{kf}\n{pose.matrix()}.')

            if i != 0:
                # preceding_constraint = g2o.Isometry3d(np.linalg.inv(self.slam_structure.poses[keyframe_ids[i-1]][0]) @ self.slam_structure.poses[kf][0])
                self.add_edge(
                    vertices=(keyframe.preceding_keyframe.idx, keyframe.idx),
                    measurement=keyframe.preceding_constraint)
                # logger.info(f'add edge:{keyframe_ids[i-1]}, {kf}\n{preceding_constraint.matrix()}.')
   
        for [kf, kf2, meas] in loops:
            meas = g2o.Isometry3d(meas)
            self.add_edge((kf, kf2), measurement=meas)
        self.propagate(anchor)
    
    def propagate(self, ref_id):
        d = max(20, len(self.slam_structure.key_frames) * 0.1)
        propagator = SmoothEstimatePropagator(self, d)
        propagator.propagate(self.vertex(ref_id))        
    
    def get_pose(self, id):
        return self.vertex(id).estimate()
    
    def update_poses_and_points(
            self, correction=None):
        for keyframe in self.slam_structure.key_frames.values():
            # intrinsics = self.slam_structure.poses[kf][1]
            intrinsics = keyframe.intrinsic
            uncorrected = keyframe.pose
            if correction is None:
                vertex = self.vertex(keyframe.idx)
                if vertex.fixed():
                    continue
                corrected = vertex.estimate()
            else:
                corrected = uncorrected * correction

            delta = uncorrected.inverse() * corrected
            if (g2o.AngleAxis(delta.rotation()).angle() < 0.02 and
                np.linalg.norm(delta.translation()) < 0.03):          # 1°, 1mm
                continue
            # self.slam_structure.poses[kf] = (corrected.matrix(), intrinsics)
            keyframe.pose = corrected
            # update mappoint
            # for point_id in keyframe.feature.keypoints_ids:
            #     mappoint = self.slam_structure.map_points[point_id]
            #     point_3d, color = mappoint.position, mappoint.color
            #     # old = point_3d
            #     if point_id in self.visitedIDs:
            #         # logger.info(f'point:{point_id} appeared before!')
            #         continue
            #     self.visitedIDs.add(point_id)
            #     old = np.vstack([point_3d[:,None], np.array([1])])
            #     # Tc@Tuc^-1@Tc@Tuc^-1
            #     # new = corrected.matrix() @ np.linalg.inv(uncorrected.matrix()) @ corrected.matrix() @ np.linalg.inv(uncorrected.matrix()) @ old
            #     new = corrected.matrix() @ (np.linalg.inv(uncorrected.matrix()) @ old)
            #     new = np.squeeze(new[:3,:])
            #     # logger.info(f'Update mappoints old: {point_3d}, new: {new}')
            #     # self.slam_structure.points[point_id] = (new, color)
            #     mappoint.update_position(new)
            #     # mappoint.set_color(color)
            for mappoint in self.slam_structure.map_points.values():
                if mappoint.id not in self.visitedIDs:
                    self.visitedIDs.add(mappoint.id)
                    point_3d = mappoint.position
                    old = np.vstack([point_3d[:,None], np.array([1])])
                    new = corrected.matrix() @ (np.linalg.inv(uncorrected.matrix()) @ old)
                    new = np.squeeze(new[:3,:])
                    mappoint.update_position(new)
            
    def run_pgo(self):
        self.optimize(20)
        self.extract_pgo_data()
        # update BA with poses and points after PGO
        # self.update_ba_data(self.BA)
        
    def extract_pgo_data(self):
        self.update_poses_and_points()