import numpy as np
import g2o

class BundleAdjuster(g2o.SparseOptimizer):
    def __init__(self, use_sparse_solver=False):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        if use_sparse_solver:
            solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

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
    
    def add_pose_edge(self, pose_indicies, measurement, 
            information=np.identity(6),
            robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(pose_indicies):
            if isinstance(v, int):
                v = self.vertex(v*2)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        
        super().add_edge(edge)

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, 
            measurement,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI

        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)

        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()


    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate()
    
    def fix_pose(self, pose_id, fixed=True):
        self.vertex(pose_id * 2).set_fixed(fixed)
    
    def fix_point(self, point_id, fixed=True):
        self.vertex(point_id * 2 + 1).set_fixed(fixed)
        
    
class LocalBA(object):
    def __init__(self, ):
        self.optimizer = BundleAdjuster()
        self.measurements = []
        self.keyframes = []
        self.mappoints = set()

        # threshold for confidence interval of 95%
        self.huber_threshold = 5.991

    def set_data(self, adjust_keyframes, fixed_keyframes):
        self.clear()
        for kf in adjust_keyframes:
            self.optimizer.add_pose(kf.id, kf.pose, kf.cam, fixed=False)
            self.keyframes.append(kf)

            for m in kf.measurements():
                pt = m.mappoint
                if pt not in self.mappoints:
                    self.optimizer.add_point(pt.id, pt.position)
                    self.mappoints.add(pt)

                edge_id = len(self.measurements)
                self.optimizer.add_edge(edge_id, pt.id, kf.id, m)
                self.measurements.append(m)

        for kf in fixed_keyframes:
            self.optimizer.add_pose(kf.id, kf.pose, kf.cam, fixed=True)
            for m in kf.measurements():
                if m.mappoint in self.mappoints:
                    edge_id = len(self.measurements)
                    self.optimizer.add_edge(edge_id, m.mappoint.id, kf.id, m)
                    self.measurements.append(m)

    def update_points(self):
        for mappoint in self.mappoints:
            mappoint.update_position(self.optimizer.get_point(mappoint.id))

    def update_poses(self):
        for keyframe in self.keyframes:
            keyframe.update_pose(self.optimizer.get_pose(keyframe.id))
            keyframe.update_reference()
            keyframe.update_preceding()

    def get_bad_measurements(self):
        bad_measurements = []
        for edge in self.optimizer.active_edges():
            if edge.chi2() > self.huber_threshold:
                bad_measurements.append(self.measurements[edge.id()])
        return bad_measurements

    def clear(self):
        self.optimizer.clear()
        self.keyframes.clear()
        self.mappoints.clear()
        self.measurements.clear()

    def abort(self):
        self.optimizer.abort()

    def optimize(self, max_iterations):
        return self.optimizer.optimize(max_iterations)


class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

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


    def set_data(self, keyframes, loops):
        super().clear()
        anchor=None
        for kf, *_ in loops:
            if anchor is None or kf < anchor:
                anchor = kf

        for i, kf in enumerate(keyframes):
            pose = g2o.Isometry3d(
                kf.orientation,
                kf.position)
            
            fixed = i == 0
            if anchor is not None:
                fixed = kf <= anchor
            self.add_vertex(kf.id, pose, fixed=fixed)

            if kf.preceding_keyframe is not None:
                self.add_edge(
                    vertices=(kf.preceding_keyframe.id, kf.id),
                    measurement=kf.preceding_constraint)

            if (kf.reference_keyframe is not None and
                kf.reference_keyframe != kf.preceding_keyframe):
                self.add_edge(
                    vertices=(kf.reference_keyframe.id, kf.id),
                    measurement=kf.reference_constraint)
        
        for kf, kf2, meas in loops:
            self.add_edge((kf.id, kf2.id), measurement=meas)


    def update_poses_and_points(
            self, keyframes, correction=None, exclude=set()):

        for kf in keyframes:
            if len(exclude) > 0 and kf in exclude:
                continue
            uncorrected = g2o.Isometry3d(kf.orientation, kf.position)
            if correction is None:
                vertex = self.vertex(kf.id)
                if vertex.fixed():
                    continue
                corrected = vertex.estimate()
            else:
                corrected = uncorrected * correction

            delta = uncorrected.inverse() * corrected
            if (g2o.AngleAxis(delta.rotation()).angle() < 0.02 and
                np.linalg.norm(delta.translation()) < 0.03):          # 1°, 3cm
                continue

            for m in kf.measurements():
                if m.from_triangulation():
                    old = m.mappoint.position
                    new = corrected * (uncorrected.inverse() * old)
                    m.mappoint.update_position(new)  
                    # update normal ?
            kf.update_pose(corrected)