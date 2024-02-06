
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from abc import ABC, abstractmethod
from collections import OrderedDict
import cv2
import numpy as np
from depth_anything.depth_anything.dpt import DepthAnything
from depth_anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet 
import models.depth.models as models


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

class DepthEstimatorDepthAnything(DepthEstimatorBase):
    def __init__(self, encoder:['vits', 'vitb', 'vitl'], multiplier=1) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(self.DEVICE).eval()
        
        self.transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
        ])
        
        self.depth = None
        self.cv_image = None
        self.save_id = 0
        
    def __call__(self, image, mask):
        image = image.permute(1, 2, 0).detach().cpu().numpy()*255
        cv_image = image.astype(np.uint8)
        self.cv_image = cv_image
        h, w = cv_image.shape[:2]
        
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) / 255.0
        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(self.DEVICE)
        # image = torch.from_numpy(image).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            depth = self.depth_anything(image)
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        self.depth = depth
        
        # depth = (depth - depth.min()) / (depth.max() - depth.min()) * self.multiplier
        # depth_cv = depth.cpu().numpy().astype(np.uint8)
        # depth_color = cv2.applyColorMap(depth_cv, cv2.COLORMAP_INFERNO)
        # combined_results = cv2.hconcat([cv_image, depth_color])
        # cv2.imwrite('depth.png', combined_results)
        # breakpoint()
        return depth

    def visualize(self):
        depth_cv = (self.depth - self.depth.min()) / (self.depth.max() - self.depth.min()) * 255.0
        depth_cv = self.depth.cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_cv, cv2.COLORMAP_INFERNO)
        combined_results = cv2.hconcat([self.cv_image, depth_color])
        cv2.imwrite(f'../datasets/temp_data/depth/{self.save_id}.png', combined_results)
        self.save_id += 1
        
# TODO: Wrapper for depth estimation network