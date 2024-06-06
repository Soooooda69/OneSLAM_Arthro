
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from abc import ABC, abstractmethod
from collections import OrderedDict
import cv2
import numpy as np
import models.depth.models as models
import statistics
from scipy.optimize import minimize
from pyntcloud import PyntCloud
import pandas as pd

def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict


class DepthEstimatorBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, image):
        return None

class DepthEstimatorConstant(DepthEstimatorBase):
    def __init__(self, multiplier=1) -> None:
        super().__init__()
        self.multiplier = multiplier

    def __call__(self, image, mask):
        return self.multiplier * torch.ones_like(image[0, ...])[None, ...]

class DepthEstimatorFCDenseNet57(DepthEstimatorBase):
    def __init__(self, trained_model_path, target_device, depth_scale_factor=1.) -> None:
        super().__init__()
        self.trained_model = models.FCDenseNet57(n_classes=1).to(target_device)
        self.trained_model.eval()
        self.trained_model.load_state_dict(remove_data_parallel(torch.load(trained_model_path)['model']))
        self.depth_scale_factor = depth_scale_factor


    def __call__(self, image, mask):
        target_shape_mult = min(image.shape[1]//256, image.shape[2]//320)
        target_shape = (256*target_shape_mult, 320*target_shape_mult)
        height_offset = (image.shape[1]-target_shape[0])//2
        width_offset = (image.shape[2]-target_shape[1])//2

        left = None if height_offset == 0 else -height_offset
        right = None if width_offset == 0 else -width_offset
            
        mask_cut = mask[:, height_offset:left, width_offset:right]
        image_cut = image[:, height_offset:left, width_offset:right]*mask_cut
        #breakpoint()



        img_cut_scaled = F.interpolate(image_cut[None], size=(256, 320), mode='bilinear')

        #breakpoint()
        with torch.no_grad():
            depth_cut_scaled = self.trained_model(img_cut_scaled)
        depth_cut = F.interpolate(depth_cut_scaled, size=target_shape, mode='bilinear')[0, ...]
        

        depth = torch.zeros_like(mask)
        depth[:, height_offset:left, width_offset:right] = depth_cut[0, ...]
        depth *= mask
        depth *= self.depth_scale_factor

        return depth

# class DepthEstimatorDepthAnything(DepthEstimatorBase):
#     def __init__(self, encoder:['vits', 'vitb', 'vitl'], multiplier=1) -> None:
#         super().__init__()
#         self.multiplier = multiplier
#         self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(self.DEVICE).eval()
#         self.mapPoint_xyzs = {}
#         self.current_frame = None
#         # self.mapPoint_xys = []
#         self.transform = Compose([
#         Resize(
#             width=518,
#             height=518,
#             resize_target=False,
#             keep_aspect_ratio=True,
#             ensure_multiple_of=14,
#             resize_method='lower_bound',
#             image_interpolation_method=cv2.INTER_CUBIC,
#         ),
#         NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         PrepareForNet(),
#         ])
        
#         self.disp = None
#         self.cv_image = None
#         self.save_id = 0
    
#     def scale_recovery(self, disp):
#         def cost(params):
#             s,t = params
#             return np.linalg.norm(s * (1/depth_anything_output)+t - mapPoint_depth)
        
#         mapPoint_depth = []
#         depth_anything_output = []
#         for keypoints_id, (keypoints, _) in self.current_frame.feature.keypoints_info.items():
#             x, y  = keypoints[0], keypoints[1]
#             depth_anything_output.append(disp[y.astype(int), x.astype(int)])
#             # transform points in world coordinate to camera coordinate
#             point3d = self.current_frame.transform(self.mapPoint_xyzs[keypoints_id][:,None])
#             mapPoint_depth.append(point3d[2])
#         depth_anything_output = np.array(depth_anything_output)
#         mapPoint_depth = np.array(mapPoint_depth).flatten()
#         s0 = statistics.median(mapPoint_depth) / (1/statistics.median(disp.flatten()))
#         t0 = 0
#         params = [s0, t0]
#         res = minimize(cost, params, method='Nelder-Mead', options={'disp': True})
#         print('Scale:', res.x[0], 'Translation:', res.x[1])
#         return res.x[0] * (1/disp) + res.x[1]
    
#     def __call__(self, image, scale_recovery=False):
#         if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
#             cv_image = image
#         else:
#             image = image.permute(1, 2, 0).detach().cpu().numpy()*255
#             cv_image = image.astype(np.uint8)
#         self.cv_image = cv_image
#         h, w = cv_image.shape[:2]
        
#         image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) / 255.0
#         image = self.transform({'image': image})['image']
#         image = torch.from_numpy(image).unsqueeze(0).to(self.DEVICE)
#         with torch.no_grad():
#             disp = self.depth_anything(image)
#         disp = F.interpolate(disp[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
#         self.disp = disp.detach().cpu().numpy()
#         # if scale_recovery:
#         #     depth = self.scale_recovery(self.disp)
#         # self.disp = (self.disp - self.disp.min()) / (self.disp.max() - self.disp.min())
#         return self.disp
    
#     def visualize(self, idx):
#         disp_cv = (self.disp - self.disp.min()) / (self.disp.max() - self.disp.min()) * 255.0
#         disp_cv = disp_cv.astype(np.uint8)
#         depth_color = cv2.applyColorMap(disp_cv, cv2.COLORMAP_INFERNO)
#         combined_results = cv2.hconcat([self.cv_image, depth_color])
        
#         cv2.imwrite(f'../datasets/temp_data/disp/{idx}.png', combined_results)
#         self.save_id += 1
    
#     def save_disp(self, frame_id):
#         # depth = self.depth.cpu().numpy()
#         np.save(f'../datasets/temp_data/disp_npy/{frame_id}.npy', self.disp)

#     def save_projection(self, frame, depth_0):
#         # test & visualize depth  
#         image = frame.feature.image
#         new_points_2d = []
#         color = []
#         for x in range(depth_0.shape[0]):
#             for y in range(depth_0.shape[1]):
#                 new_points_2d.append([x, y])
#                 color.append(image[y, x])
#         # for x, y in self.mapPoint_xys:
#                 # new_points_2d.append([x.astype(int), y.astype(int)])
#                 # color.append(image[x.astype(int), y.astype(int)])
                
#         new_points_2d = np.vstack(new_points_2d)
#         color = np.vstack(color).astype(np.uint8)
#         new_points_3d = frame.unproject(new_points_2d, depth_0)
        
#         point_cloud = pd.DataFrame({
#             'x': new_points_3d[:, 0],
#             'y': new_points_3d[:, 1],
#             'z': new_points_3d[:, 2],
#             'red': color[:, 0],
#             'green': color[:, 1],
#             'blue': color[:, 2]
#         })
#         pynt_cloud = PyntCloud(point_cloud)
#         pynt_cloud.to_file('preeeeeeeeeeeeeeeeeeeeed.ply')

# # TODO: Wrapper for depth estimation network
