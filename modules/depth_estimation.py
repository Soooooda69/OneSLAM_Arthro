
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from collections import OrderedDict


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
    def __init__(self, multiplier=10) -> None:
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



# TODO: Wrapper for depth estimation network