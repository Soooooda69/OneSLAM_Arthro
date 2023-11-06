import numpy as np
from pathlib import Path
from skimage import measure
import pickle
import torch
from torch import nn
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets.dataset import ImageDataset
from tqdm import tqdm
import models.depth.models as models
from collections import OrderedDict
from modules.depth_estimation import *
from datasets.transforms import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import cv2
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
except Exception as err:
    print('Warning: %s' % (str(err)))
    print('Failed to import PyCUDA.')
    exit()

import argparse
import torch.backends.cudnn as cudnn
import random


class DepthScalingLayer(nn.Module):
    def __init__(self, epsilon=1.0e-8):
        super(DepthScalingLayer, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()
        self.zero = torch.tensor(0.0).float().cuda()
        self.one = torch.tensor(1.0).float().cuda()

    def forward(self, x):
        absolute_depth_estimations, input_sparse_depths, input_weighted_sparse_masks, intrinsics = x
        # Use sparse depth values which are greater than a certain ratio of the mean value of the sparse depths to avoid
        # unstability of scale recovery
        input_sparse_binary_masks = torch.where(input_weighted_sparse_masks > 1.0e-8, self.one, self.zero)
        mean_sparse_depths = torch.sum(input_sparse_depths * input_sparse_binary_masks, dim=(1, 2, 3),
                                       keepdim=True) / torch.sum(input_sparse_binary_masks, dim=(1, 2, 3), keepdim=True)
        above_mean_masks = torch.where(input_sparse_depths > 0.5 * mean_sparse_depths, self.one, self.zero)

        # Introduce a criteria to reduce the variation of scale maps
        sparse_scale_maps = input_sparse_depths * above_mean_masks / (self.epsilon + absolute_depth_estimations)
        mean_scales = torch.sum(sparse_scale_maps, dim=(1, 2, 3), keepdim=True) / torch.sum(above_mean_masks,
                                                                                            dim=(1, 2, 3), keepdim=True)
        centered_sparse_scale_maps = sparse_scale_maps - above_mean_masks * mean_scales
        scale_stds = torch.sqrt(torch.sum(centered_sparse_scale_maps * centered_sparse_scale_maps, dim=(1, 2, 3),
                                          keepdim=False) / torch.sum(above_mean_masks, dim=(1, 2, 3), keepdim=False))
        scales = torch.sum(sparse_scale_maps, dim=(1, 2, 3)) / torch.sum(above_mean_masks, dim=(1, 2, 3))

        return torch.mul(scales.reshape(-1, 1, 1, 1), absolute_depth_estimations), torch.mean(scale_stds / mean_scales)

class DepthScalingLayer3D(nn.Module):
    def __init__(self, epsilon=1.0e-8, bb_scale_type="MinMax"):
        super(DepthScalingLayer3D, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()
        self.zero = torch.tensor(0.0).float().cuda()
        self.one = torch.tensor(1.0).float().cuda()

        assert bb_scale_type in ["MinMax", "Min", "Max"]
        self.bb_scale_type = bb_scale_type

    def unproject(self, point_2d, depth, intrinsics):
        x_d, y_d = point_2d[0], point_2d[1]
        fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

        x = ((x_d - cx) * depth / fx)
        y = ((y_d - cy) * depth / fy)
        z = depth

        point_3d = np.stack([x, y, z])

        return point_3d
    
    def find_scale_to_unit(self, points):
        for i, p in enumerate(points):
            if i == 0:
                max_bound = np.asarray(p, dtype=np.float32)
                min_bound = np.asarray(p, dtype=np.float32)
            else:
                temp = np.asarray(p, dtype=np.float32)
                if np.any(np.isnan(temp)):
                    continue
                max_bound = np.maximum(max_bound, temp)
                min_bound = np.minimum(min_bound, temp)
        
        if self.bb_scale_type == "MinMax":
            return np.linalg.norm(max_bound-min_bound, ord=2)
        if self.bb_scale_type == "Min":
            return np.linalg.norm(min_bound, ord=2)
        if self.bb_scale_type == "Max":
            return np.linalg.norm(max_bound, ord=2)
    
        return None

    def forward(self, x):
        depth_estimations, sparse_depths, sparse_depth_masks, intrinsics = x
        # Use sparse depth values which are greater than a certain ratio of the mean value of the sparse depths to avoid
        # unstability of scale recovery
        points_2d = []
        points_2d_torch = (sparse_depth_masks != 0).nonzero() # NOTE: y,x order
        for i in range(len(points_2d_torch)):
            points_2d.append(points_2d_torch[i][2:].detach().cpu().numpy()[::-1])

        points_3d_sparse = []
        points_3d_dense = []

        for i in range(len(points_2d_torch)):
            points_3d_sparse.append(self.unproject(points_2d[i], sparse_depths[0, 0, points_2d[i][1], points_2d[i][0]].item(), intrinsics))
            points_3d_dense.append(self.unproject(points_2d[i], depth_estimations[0, 0, points_2d[i][1], points_2d[i][0]].item(), intrinsics))
        
        scale_sparse = self.find_scale_to_unit(points_3d_sparse)
        scale_dense = self.find_scale_to_unit(points_3d_dense)


        rescaled_depth = (scale_sparse / scale_dense) * depth_estimations
        std = torch.tensor(1.).to(depth_estimations.device) # Just some value


        print(f"Scale: {scale_sparse / scale_dense}")
        return rescaled_depth, std


class TSDFVolume(object):
    def __init__(self, vol_bnds, voxel_size, trunc_margin):
        # Define voxel volume parameters
        self._vol_bnds = vol_bnds  # rows: x,y,z columns: min,max in world coordinates in meters
        self._voxel_size = voxel_size
        self._trunc_margin = trunc_margin

        # Adjust volume bounds
        self._vol_dim = np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(
            order='C').astype(int)  # ensure C-order contiguous
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + \
            self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(
            order='C').astype(np.float32)  # ensure C-order contiguous
        print("Voxel volume size: %d x %d x %d" %
              (self._vol_dim[0], self._vol_dim[1], self._vol_dim[2]))

        # Initialize pointers to voxel volume in CPU memory
        # Assign oversized tsdf volume
        self._tsdf_vol_cpu = np.zeros(
            self._vol_dim).astype(np.float32)  # -2.0 *
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(
            np.float32)
        self._uncertainty_vol_cpu = np.zeros(self._vol_dim).astype(
            np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        # Copy voxel volumes to GPU
        self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
        cuda.memcpy_htod(self._tsdf_vol_gpu, self._tsdf_vol_cpu)
        self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
        cuda.memcpy_htod(self._weight_vol_gpu, self._weight_vol_cpu)
        self._uncertainty_vol_gpu = cuda.mem_alloc(
            self._uncertainty_vol_cpu.nbytes)
        cuda.memcpy_htod(self._uncertainty_vol_gpu, self._uncertainty_vol_cpu)
        self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
        cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)

        # Cuda kernel function (C++)
        self._cuda_src_mod_with_confidence_map = SourceModule("""
          __global__ void integrate(float * tsdf_vol,
                                    float * weight_vol,
                                    float * uncertainty_vol,
                                    float * color_vol,
                                    float * vol_dim,
                                    float * vol_origin,
                                    float * cam_intr,
                                    float * cam_pose,
                                    float * other_params,
                                    float * color_im,
                                    float * depth_im,
                                    float * std_im) {

            // Get voxel index
            int gpu_loop_idx = (int) other_params[0];
            int max_threads_per_block = blockDim.x;
            int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
            int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;

            int vol_dim_x = (int) vol_dim[0];
            int vol_dim_y = (int) vol_dim[1];
            int vol_dim_z = (int) vol_dim[2];

            if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
                return;

            // Get voxel grid coordinates (note: be careful when casting)
            float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
            float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
            float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);

            // Voxel grid coordinates to world coordinates
            float voxel_size = other_params[1];
            float pt_x = vol_origin[0]+voxel_x*voxel_size;
            float pt_y = vol_origin[1]+voxel_y*voxel_size;
            float pt_z = vol_origin[2]+voxel_z*voxel_size;

            // World coordinates to camera coordinates
            float tmp_pt_x = pt_x-cam_pose[0*4+3];
            float tmp_pt_y = pt_y-cam_pose[1*4+3];
            float tmp_pt_z = pt_z-cam_pose[2*4+3];
            float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
            float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
            float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;

            // Because of the long tube of endoscope, the minimum depth to consider is not zero
            float min_depth = other_params[6];
            if (cam_pt_z < min_depth) {
                return;
            }
                                                                                                     
            
            // Camera coordinates to image pixels
            int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
            int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);

            // Skip if outside view frustum
            int im_h = (int) other_params[2];
            int im_w = (int) other_params[3];
            if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h)
                return;

            // Skip invalid depth
            float depth_value = depth_im[pixel_y*im_w+pixel_x];
            float std_value = std_im[pixel_y*im_w + pixel_x];
            if (depth_value <= 0 || std_value <= 0) {
                return;
            }

            // Get std value for the current observation
            float trunc_margin = other_params[4];
            float depth_diff = depth_value - cam_pt_z;
            if (depth_diff < -trunc_margin)
                return;

            float dist = fmin(1.0f, depth_diff / std_value);

            float w_old = weight_vol[voxel_idx];
            float obs_weight = other_params[5];
            float w_new = w_old + obs_weight;
            tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx] * w_old + dist * obs_weight) / w_new;
            weight_vol[voxel_idx] = w_new;

            
            // Integrate color
            float new_color = color_im[pixel_y * im_w + pixel_x];
            float new_b = floorf(new_color / (256 * 256));
            float new_g = floorf((new_color - new_b * 256 * 256) / 256);
            float new_r = new_color - new_b * 256 * 256 - new_g * 256;

            float old_color = color_vol[voxel_idx];
            float old_b = floorf(old_color / (256 * 256));
            float old_g = floorf((old_color - old_b * 256 * 256) / 256);
            float old_r = old_color - old_b * 256 * 256 - old_g * 256;

            new_b = fmin(roundf((old_b * w_old + new_b * obs_weight) / w_new), 255.0f);
            new_g = fmin(roundf((old_g * w_old + new_g * obs_weight) / w_new), 255.0f);
            new_r = fmin(roundf((old_r * w_old + new_r * obs_weight) / w_new), 255.0f);

            color_vol[voxel_idx] = new_b * 256 * 256 + new_g * 256 + new_r;
          }""")

        self._cuda_integrate = self._cuda_src_mod_with_confidence_map.get_function(
            "integrate")
        # Determine block/grid size on GPU
        gpu_dev = cuda.Device(0)
        self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
        n_blocks = int(np.ceil(float(np.prod(self._vol_dim)) /
                       float(self._max_gpu_threads_per_block)))
        grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X,
                         int(np.floor(np.cbrt(n_blocks))))
        grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y, int(
            np.floor(np.sqrt(n_blocks / grid_dim_x))))
        grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z, int(
            np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))))
        self._max_gpu_grid_dim = np.array(
            [grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
        # _n_gpu_loops specifies how many loops for the GPU to process the entire volume
        self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim)) / float(
            np.prod(self._max_gpu_grid_dim) * self._max_gpu_threads_per_block)))

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, min_depth, std_im, obs_weight=1.):
        im_h = depth_im.shape[0]
        im_w = depth_im.shape[1]

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(
            color_im[:, :, 2] * 256 * 256 + color_im[:, :, 1] * 256 + color_im[:, :, 0])


        # integrate voxel volume (calls CUDA kernel)
        for gpu_loop_idx in range(self._n_gpu_loops):
            self._cuda_integrate(self._tsdf_vol_gpu,
                                 self._weight_vol_gpu,
                                 self._uncertainty_vol_gpu,
                                 self._color_vol_gpu,
                                 cuda.InOut(self._vol_dim.astype(np.float32)),
                                 cuda.InOut(
                                     self._vol_origin.astype(np.float32)),
                                 cuda.InOut(
                                     cam_intr.reshape(-1).astype(np.float32)),
                                 cuda.InOut(
                                     cam_pose.reshape(-1).astype(np.float32)),
                                 cuda.InOut(np.asarray(
                                     [gpu_loop_idx, self._voxel_size, im_h, im_w, self._trunc_margin,
                                      obs_weight, min_depth],
                                     np.float32)),
                                 cuda.InOut(
                                     color_im.reshape(-1).astype(np.float32)),
                                 cuda.InOut(
                                     depth_im.reshape(-1).astype(np.float32)),
                                 cuda.InOut(
                                     std_im.reshape(-1).astype(np.float32)),
                                 block=(self._max_gpu_threads_per_block, 1, 1), grid=(
                                     int(self._max_gpu_grid_dim[0]), int(
                                         self._max_gpu_grid_dim[1]),
                                     int(self._max_gpu_grid_dim[2])))

    # Copy voxel volume to CPU
    def get_volume(self):
        cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
        cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
        cuda.memcpy_dtoh(self._weight_vol_cpu, self._weight_vol_gpu)
        return self._tsdf_vol_cpu, self._color_vol_cpu, self._weight_vol_cpu

    # Get mesh of voxel volume via marching cubes
    def get_mesh(self, only_visited=False):
        tsdf_vol, color_vol, weight_vol = self.get_volume()

        verts, faces, norms, vals = measure.marching_cubes(
            tsdf_vol, level=0, gradient_direction='ascent')

        verts_ind = np.round(verts).astype(int)
        # voxel grid coordinates to world coordinates
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        std_vals = weight_vol[verts_ind[:, 0],
                              verts_ind[:, 1], verts_ind[:, 2]]
        std_vals = np.uint8(std_vals / np.max(std_vals) * 255)
        std_colors = std_vals.astype(np.uint8).reshape(-1, 1)
        std_colors = cv2.cvtColor(cv2.applyColorMap(
            std_colors, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB).reshape(-1, 3)

        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / (256 * 256))
        colors_g = np.floor((rgb_vals - colors_b * 256 * 256) / 256)
        colors_r = rgb_vals - colors_b * 256 * 256 - colors_g * 256
        colors = np.transpose(
            np.uint8(np.floor(np.asarray([colors_r, colors_g, colors_b])))).reshape(-1, 3)

        if only_visited:
            verts_indxes = verts_ind[:, 0] * weight_vol.shape[1] * weight_vol.shape[2] + \
                verts_ind[:, 1] * weight_vol.shape[2] + verts_ind[:, 2]
            weight_vol = weight_vol.reshape((-1))
            valid_vert_indexes = np.nonzero(weight_vol[verts_indxes] >= 1)[0]
            valid_vert_indexes = set(valid_vert_indexes)

            indicators = np.array([face in valid_vert_indexes for face in faces[:, 0]]) \
                & np.array([face in valid_vert_indexes for face in faces[:, 1]]) \
                & np.array([face in valid_vert_indexes for face in faces[:, 2]])

            return verts, faces[indicators], norms, colors, std_colors

        return verts, faces, norms, colors, std_colors

# Get corners of 3D camera view frustum of depth image
def get_view_frustum(depth_im, cam_intr, cam_pose):
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([(np.array([0, 0, 0, im_w, im_w]) - cam_intr[2]) * np.array(
        [0, max_depth, max_depth, max_depth, max_depth]) / cam_intr[0],
        (np.array([0, 0, im_h, 0, im_h]) - cam_intr[3]) * np.array(
        [0, max_depth, max_depth, max_depth, max_depth]) / cam_intr[1],
        np.array([0, max_depth, max_depth, max_depth, max_depth])])
    view_frust_pts = np.dot(cam_pose[:3, :3], view_frust_pts) + np.tile(cam_pose[:3, 3].reshape(3, 1), (
        1, view_frust_pts.shape[1]))  # from camera to world coordinates
    return view_frust_pts

def meshwrite(filename, verts, faces, norms, colors):
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
            verts[i, 0], verts[i, 1], verts[i, 2], norms[i, 0], norms[i,
                                                                      1], norms[i, 2], colors[i, 0], colors[i, 1],
            colors[i, 2]))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" %
                       (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()

def get_sparse_depths(frame_idx, pose, intrinsics, points, pose_point_map, mask):
    # TODO: Return sparse depth map and sparse depth mask as torch tensor
    
    pose_points = pose_point_map[frame_idx]

    
    intrinsics_mat = np.identity(3)
    intrinsics_mat[0, 0] = intrinsics[0]
    intrinsics_mat[1, 1] = intrinsics[1]
    intrinsics_mat[0, 2] = intrinsics[2]
    intrinsics_mat[1, 2] = intrinsics[3]

    world_to_cam = np.linalg.inv(pose)
    projection_mat = np.identity(4)[:3]

    sparse_depths = torch.zeros_like(mask)
    sparse_depths_mask = torch.zeros_like(mask)
    
    for (point_id, point_2d) in pose_points:
        if point_id not in points.keys():
            continue
        point_3d, _ = points[point_id]
        point_3d_homogenous = np.append(point_3d, 1)

        # Check if point properly projects
        point_2d_projected = intrinsics_mat @ projection_mat @ world_to_cam @ point_3d_homogenous
        d, point_2d_projected = point_2d_projected[2], point_2d_projected[:2]/(1e-6 + point_2d_projected[2])
        
        # TODO: Filter out points with high reprojection error (should already be filtered)
        # TODO: point_2d or point_2d_projected?
        #if np.linalg.norm(point_2d - point_2d_projected) > 5:
        #    continue

        #print(np.linalg.norm(point_2d - point_2d_projected))
        try:
            sparse_depths[0, int(point_2d_projected[1]), int(point_2d_projected[0])] = d
            sparse_depths_mask[0, int(point_2d_projected[1]), int(point_2d_projected[0])] = 1
        except:
            print("Error when trying to set sparse depth.")
            #breakpoint()

    sparse_depths *= mask
    sparse_depths_mask *= mask

    return sparse_depths, sparse_depths_mask


# Start of depth fusion
parser = argparse.ArgumentParser(description='Python file for fusing dense meshes',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data loading
parser.add_argument("--exp_root", required=True, type=str, help="path to dataset")
parser.add_argument("--process_subset", action='store_true', help="use start_idx/end_idx/image_subsample to define a subset of processed keyframes")
parser.add_argument("--start_idx", default=-1, type=int, help="starting frame index for processing")
parser.add_argument("--end_idx", default=-1, type=int, help="end frame index for processing")
parser.add_argument("--img_width", default=-1, type=int, help="width to rescale images to, -1 for no scaling")
parser.add_argument("--img_height", default=-1, type=int, help="height to rescale images to, -1 for no scaling")
parser.add_argument("--lumen_mask_low_threshold", default=0., type=float, help="mask out pixels that are too dark")
parser.add_argument("--lumen_mask_high_threshold", default=1., type=float, help="mask out pixels that are too bright")

# Depths
parser.add_argument("--depth_network_path", default="./trained_models/depth/depth_estimator_8_sequences_epoch_200.pt", type=str, help="path to depth network, only required for depth estimation using the network")
parser.add_argument("--depth_scaling", type=str, choices=['xingtong', '3d_min_max', '3d_min', '3d_max'], default='3d_min', 
                    help='how to sample new points at each section start'
                         'xingtong: xingtongs scaling method'
                         '3d_min_max: scale according to minimum and maximum of 3d bounding box'
                         '3d_min: scale according to minimum of 3d bounding box'
                         '3d_max: scale according to maximum of 3d bounding box')

# TSDF Fusion
parser.add_argument("--voxel_size", default=0.1, type=float, help="size of a voxel")
parser.add_argument("--max_voxel_count", default=400.0 ** 3, type=float, help="maximum number of voxels")
parser.add_argument("--trunc_margin_multiplier", default=10., type=float, help="trunc margin")



# General Stuff
parser.add_argument("--target_device", default='cuda:0', type=str, help="GPU to run process on")
parser.add_argument("--mesh_name", default='fused_mesh.ply', type=str, help="name of outputted mesh")
parser.add_argument("--seed", default=42, type=int, help="seed for reproducability")


# Parse arguments
args = parser.parse_args()

# If processing subset, start_idx, end_idx and image_subsample must be set.
if args.process_subset:
    assert args.start_idx != -1
    assert args.end_idx != -1

# Set seed for reproduceability
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
cudnn.deterministic = True

# Load data
exp_root = Path(args.exp_root)

with open(exp_root / 'keyframes.pickle', 'rb') as handle:
    keyframes = pickle.load(handle)

with open(exp_root / 'poses.pickle', 'rb') as handle:
    poses = pickle.load(handle)

with open(exp_root / 'points.pickle', 'rb') as handle:
    points = pickle.load(handle)

with open(exp_root / 'pose_point_map.pickle', 'rb') as handle:
    pose_point_map = pickle.load(handle)

data_root = exp_root / 'dataset'
composed_transforms = transforms.Compose([SampleToTensor(),
                                          RescaleImages((args.img_height, args.img_width)),
                                          MaskOutLuminosity(threshold_high=args.lumen_mask_high_threshold, threshold_low=args.lumen_mask_low_threshold),
                                          SampleToDevice(args.target_device)])
dataset = ImageDataset(data_root, transform=composed_transforms)

# Determine frames to process
frames_to_process = list(keyframes)
if args.process_subset:
    frames_to_process = list(filter(lambda frame_idx: frame_idx in range(args.start_idx, args.end_idx), frames_to_process))


# Load depth related stuff
depth_estimator = DepthEstimatorFCDenseNet57(args.depth_network_path, args.target_device)

if args.depth_scaling == 'xingtong':
    depth_scaler = DepthScalingLayer().to(args.target_device)
elif args.depth_scaling == '3d_min_max':
    depth_scaler = DepthScalingLayer3D(bb_scale_type="MinMax").to(args.target_device)
elif args.depth_scaling == '3d_min':
    depth_scaler = DepthScalingLayer3D(bb_scale_type="Min").to(args.target_device)
elif args.depth_scaling == '3d_max':
    depth_scaler = DepthScalingLayer3D(bb_scale_type="Max").to(args.target_device)
else:
    raise ValueError(f'Unknown argument for --depth_scaling: {args.depth_scaling}')

# Compute camera view frustum and extend convex hull
print("Estimating volume bounds ...")
vol_bnds = np.zeros((3, 2))
for keyframe in tqdm(frames_to_process):
    sample = dataset[keyframe]
    pose, intrinsics = poses[keyframe]
    
    image = sample['image']
    mask = sample['mask']
    sparse_depth, sparse_depth_mask = get_sparse_depths(keyframe, pose, intrinsics, points, pose_point_map, mask)

    pred_depth = depth_estimator(image, mask)[None]
    sparse_depth = sparse_depth[None]
    sparse_depth_mask = sparse_depth_mask[None]
    sparse_depth_mask[pred_depth == 0] = 0
    sparse_depth[pred_depth == 0] = 0

    rescaled_depth_map, std = depth_scaler([pred_depth, sparse_depth, sparse_depth_mask, intrinsics])
    
    rescaled_depth_map = rescaled_depth_map.detach().cpu().numpy().squeeze()
    
    print(rescaled_depth_map.max(), std)
    #pose = np.linalg.inv(pose)
    view_frust_pts = get_view_frustum(rescaled_depth_map, intrinsics, pose)
    vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))


print("Setting up volume ...")
voxel_size = args.voxel_size
vol_dim = (vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size
# Adaptively change the size of one voxel to fit into the GPU memory
volume = vol_dim[0] * vol_dim[1] * vol_dim[2]

max_voxel_count = args.max_voxel_count
trunc_margin_multiplier = args.trunc_margin_multiplier
factor = (volume / max_voxel_count) ** (1.0 / 3.0)
voxel_size *= factor
print("Voxel size: {}".format(voxel_size))

print("Integrating depth images...")
# Initialize voxel volume
tsdf_vol = TSDFVolume(vol_bnds, voxel_size=voxel_size,
                     trunc_margin=voxel_size * trunc_margin_multiplier)


print("Integrating depth images...")
# Only use keyframes for now
for frame_idx in tqdm(frames_to_process):
    sample = dataset[frame_idx]
    pose, _ = poses[frame_idx]

    # Get intrinsics
    intrinsics = sample['intrinsics'].cpu().numpy()
    intrinsics_mat = np.identity(3)
    intrinsics_mat[0, 0] = intrinsics[0]
    intrinsics_mat[1, 1] = intrinsics[1]
    intrinsics_mat[0, 2] = intrinsics[2]
    intrinsics_mat[1, 2] = intrinsics[3]

    # Get depth data
    image = sample['image']
    mask = sample['mask']
    sparse_depth, sparse_depth_mask = get_sparse_depths(frame_idx, pose, intrinsics, points, pose_point_map, mask)

    pred_depth = depth_estimator(image, mask)[None]
    sparse_depth = sparse_depth[None]
    sparse_depth_mask = sparse_depth_mask[None]
    sparse_depth_mask[pred_depth == 0] = 0
    sparse_depth[pred_depth == 0] = 0

    rescaled_depth_map, std = depth_scaler([pred_depth, sparse_depth, sparse_depth_mask, intrinsics])

    rescaled_depth_map = rescaled_depth_map.detach().cpu().numpy().squeeze()
    
    
    valid_mask = (rescaled_depth_map > 0.0).astype(np.float32)
    
    
    std_map = np.ones_like(rescaled_depth_map) * valid_mask

    assert not torch.isnan(std)

    # Move image to cpu
    image_np = 255*image.permute(1, 2, 0).cpu().numpy()

    tsdf_vol.integrate(image_np, rescaled_depth_map, intrinsics_mat, pose, min_depth=1.0e-4, std_im=std_map, obs_weight=1.)


print("Writing mesh model...")
verts, faces, norms, colors, _ = tsdf_vol.get_mesh(only_visited=True)
fused_model_path = str(exp_root / args.mesh_name)
meshwrite(fused_model_path, verts, faces, -norms, colors)
print("Done.")
