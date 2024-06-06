import numpy as np
from PIL import Image
import cv2 as cv
import torch
from torch.utils.data import Dataset

from pathlib import Path
import json
from natsort import natsorted
from misc.tum_tools import read_trajectory

class ImageDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = Path(data_root)

        # Required data for predictions
        self.images = self.acquire_images()
        try:
            self.frame_masks = self.acquire_frame_masks()
        except:
            self.frame_masks = None
            self.mask = self.acquire_mask()
        self.cam_calibration, self.cam_distortion = self.acquire_cam_calibration()
        self.timestamps = self.acquire_timestamps()
        # Optional ground truth
        self.depths, self.depths_available = self.acquire_depths()
        self.poses, self.poses_available = self.acquire_poses()

        self.num_frames = len(self.images)

        #if self.depths_available:
        #    assert self.num_frames == len(self.depths_available)

        #if self.poses_available:
        #    assert self.num_frames == len(self.depths_available)

        self.transform = transform

    def acquire_timestamps(self):
        time_path = self.data_root / "timestamp.txt"
        #breakpoint()
        assert time_path.exists()
        assert time_path.is_file()

        timestamp_dict = dict()
        
        with open(time_path) as file:
            for frame_idx, line in enumerate(file):
                timestamp = line
                timestamp_dict[int(frame_idx)] = float(timestamp)/1e9
        return timestamp_dict
    
    def acquire_images(self):
        image_path = self.data_root / "images"
        if not image_path.exists():
            image_path = self.data_root / "undistort_images"
        #breakpoint()
        assert image_path.exists()
        assert image_path.is_dir()

        image_path_list = natsorted(list(image_path.glob('*.jpg')) + list(image_path.glob('*.png')))
        image_dict = dict()
        count = 0
        for path in image_path_list:
            # image_dict[int(str(path)[-12:-4])] = path
            image_dict[count] = path
            count += 1
        return image_dict
    
    def acquire_cam_calibration(self):
        cam_calibration_file = self.data_root / "cam_params" / "calibration.json"
        cam_distortion_file = self.data_root / "cam_params" / "distortion_coeff.json"
        assert cam_calibration_file.exists()
        assert cam_calibration_file.is_file()
        assert cam_distortion_file.exists()
        assert cam_distortion_file.is_file()
        
        cam_calibration_data = None
        cam_distortion_data = None
        with open(cam_calibration_file) as file, open(cam_distortion_file) as distortion_file:
            cam_calibration_data = json.load(file)
            cam_distortion_data = json.load(distortion_file)
            
        return cam_calibration_data, cam_distortion_data
    
    def acquire_mask(self):
        mask_path = self.data_root / "mask.bmp"
        if not mask_path.exists() or not mask_path.is_file():
            return None
        
        mask = np.array(Image.open(mask_path))
        mask[mask < 255] = 0
        mask[mask == 255] = 255

        if len(mask.shape) > 2:
            mask = mask[..., 0]

        return mask[..., None]
    
    def acquire_frame_masks(self):
        image_path = self.data_root / "masks"
        #breakpoint()
        assert image_path.exists()
        assert image_path.is_dir()

        image_path_list = natsorted(list(image_path.glob('*.jpg')) + list(image_path.glob('*.png')))
        mask_dict = dict()
        count = 0
        for path in image_path_list:
            # image_dict[int(str(path)[-12:-4])] = path
            mask_dict[count] = path
            count += 1
        return mask_dict
    
    def acquire_depths(self):
        depth_path = self.data_root / "depths"
        if not depth_path.exists() or not depth_path.is_dir():
            return dict(), False

        depth_path_list = sorted(list(depth_path.glob('*.png')))

        depth_dict = dict()
        for path in depth_path_list:
            depth_dict[int(str(path)[-12:-4])] = path

        return depth_dict, True
        
    def acquire_poses(self):
        pose_path = self.data_root / "poses_gt.txt"
        if not pose_path.exists() or not pose_path.is_file():
            return dict(), False

        return read_trajectory(str(pose_path), matrix=True), True

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        assert idx in self.images.keys()
        # Load data in numpy format
        image = np.array(Image.open(self.images[idx]))
        depth = np.zeros_like(image)[..., 0][..., None]
        pose = np.identity(4)
        if self.frame_masks is not None:
            mask = np.array(Image.open(self.frame_masks[idx]))[..., 0][..., None]
        else:
            mask = 255*np.ones_like(image)[..., 0][..., None] if self.mask is None else self.mask

        timestamp = self.timestamps[idx]

        if self.depths_available and idx in self.depths.keys():
            depth = np.asarray(cv.imread(str(self.depths[idx]), cv.IMREAD_UNCHANGED)).astype(np.float32)[..., None]
            depth /= 1000.
            depth[mask != 255] = 0

        if self.poses_available:
            pose = self.poses[idx]


        intrinisics = self.cam_calibration['intrinsics']
        distortion = self.cam_distortion
        intrinisics_arr = np.array([intrinisics['fx'], intrinisics['fy'], intrinisics['cx'], intrinisics['cy']])
        distortion = np.array([distortion['k1'], distortion['k2'], distortion['p1'], distortion['p2'], distortion['k3']])
        sample = {
            'frame_idx': idx,
            'timestamp': timestamp,
            'image':image,
            'intrinsics':intrinisics_arr,
            'mask':mask,
            'pose':pose,
            'depth': depth,
            'distortion': distortion
        }

        if self.transform:
            sample = self.transform(sample)

        return sample