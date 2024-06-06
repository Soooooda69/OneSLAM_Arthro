from natsort import natsorted
from tqdm import tqdm
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R
import argparse
import re

class Helper:
    def __init__(self) -> None:
        pass
    
    def save_trajectory(self, trans_data, timestamps, output_file_path):
        # trans_data: Nx7
        # Write rows to the text file
        with open(output_file_path, 'w') as file:
            for time, row in zip(timestamps, trans_data):
                file.write(time + ' '+ str(row[0]) + ' ' + str(row[1]) + ' '
                           + str(row[2]) + ' '+ str(row[3]) + ' '
                           + str(row[4]) + ' '+ str(row[5]) + ' '
                           + str(row[6]) + '\n')
                # file.write(time + ', ' + str(row) +'\n')
    
    
    def make_7DOF(self, R_mat, t):
        r = R.from_matrix(R_mat)
        quat = r.as_quat()
        return np.hstack((t.T, quat[None, :]))
        

    def make_SE3_matrix(self, R, t):
        ## R: 3x3, t: 3x1
        trans_matrix = np.identity(4)
        trans_matrix[:3, :3] = R
        trans_matrix[:3, 3] = t.reshape(3,)
        return trans_matrix
    
    def inverse_sim3(self, rotation_matrix, translation_vector, scale_correction):
        # Invert the scale correction
        scale_inv = 1 / scale_correction
        # Invert the rotation matrix
        rotation_inv = rotation_matrix.T * scale_inv
        # Invert the translation vector
        translation_inv = -scale_inv * np.dot(rotation_inv, translation_vector)
        # Construct the inverse Sim(3) matrix
        sim3_inv = np.eye(4)
        sim3_inv[0:3, 0:3] = scale_inv * rotation_inv
        sim3_inv[0:3, 3] = translation_inv
        return sim3_inv
    
    def rectify_image(self, image, intrinsic_matrix, distortion_coefficients):

        # Get image size
        h, w = image.shape[:2]
        # Undistort the image
        undistorted_image = cv2.undistort(image, intrinsic_matrix, distortion_coefficients, None)

        # Rectify the image
        map_x, map_y = cv2.initUndistortRectifyMap(intrinsic_matrix, distortion_coefficients, None, None, (w, h), 5)
        rectified_image = cv2.remap(undistorted_image, map_x, map_y, cv2.INTER_LINEAR)
        
        return rectified_image
    
    
    def make_video(self, image_dir, out_name, frame_rate):

        # Output video file name
        output_video = os.path.join(out_name)

        # Get the list of image files in the directory
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

        # Sort the image files based on their names (assuming they are named in order)
        image_files = natsorted(image_files)

        # Get the first image to determine the frame size
        first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
        frame_size = (first_image.shape[1], first_image.shape[0])

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
        out = cv2.VideoWriter(output_video, fourcc, frame_rate, frame_size)

        # Iterate through the image files and write each frame to the video
        for i, image_file in tqdm(enumerate(image_files)):
            image_path = os.path.join(image_dir, image_file)
            frame = cv2.imread(image_path)
            out.write(frame)

        # Release the VideoWriter
        out.release()

    def read_trajectory(self, file_path):
        poses = {}
        with open(file_path, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line = line.strip().split(' ')
                if len(line) > 0 and line[0] != "#":
                    time = line[0]
                    t = np.array([float(x) for x in line[1:4]])
                    q = np.array([float(x) for x in line[4:]])
                    Rot = R.from_quat(q).as_matrix()
                    trans_mat = self.make_SE3_matrix(Rot, t)
                    poses[time] = trans_mat
        return poses
                    
    def align_traj(self, root_path, file_path):
        def extract_matrix(text):
            matrix_regex = r"\[(.+?)\]"
            rows = re.findall(matrix_regex, text, re.DOTALL)
            matrix = []
            for row in rows:
                values = re.findall(r"-?\d+\.\d+", row)
                matrix.append([float(val) for val in values])
            return np.array(matrix)

        def extract_scale(text):
            scale_regex = r"Scale correction: (-?\d+\.\d+)"
            scale = re.search(scale_regex, text).group(1)
            return float(scale)

        # Read the text file
        with open(os.path.join(root_path, "evo_align_res.txt"), "r") as file:
            text = file.read()

        # Extract the rotation matrix and translation vector
        rotation_text = re.search(r"Rotation of alignment:\n(.*?)\n\n", text, re.DOTALL).group(1)
        tmp_mat = extract_matrix(rotation_text)
        # Extract the scale correction
        translation_vector = tmp_mat[3, :3]
        scale_correction = extract_scale(text)
        rotation_matrix = tmp_mat[:3, :3] * scale_correction
        simTransform = np.eye(4)
        simTransform[:3, :3] = rotation_matrix
        simTransform[:3, 3] = translation_vector
        # itransform = np.linalg.inv(simTransform)
        print(np.linalg.inv(simTransform))
        
        save_path = os.path.join(root_path, 'aligned_poses_gt.txt')
        poses = self.read_trajectory(file_path) # GT poses
        itransform = self.inverse_sim3(rotation_matrix, translation_vector, scale_correction)
        print(itransform)
        with open(save_path, 'w') as file:
            for time, pose in poses.items():
                aligned_pose = itransform @ pose
                print(aligned_pose[:3, 3],R.from_matrix(aligned_pose[:3, :3]).as_quat())
                tmp = np.hstack((aligned_pose[:3, 3], R.from_matrix(aligned_pose[:3, :3]).as_quat()))
                file.write(' '.join([time] + [str(x) for x in tmp]) + '\n')
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Helper functions for OneSLAM',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--track_path", default='../datasets/temp_data/localize_tracking', type=str, help="path to save the tracking output")
    parser.add_argument("--traj_path", default=' ', type=str, help="path to trajectory file needs to be aligned")
    parser.add_argument("--root_path", default=' ', type=str, help="path to experiment root directory")
    parser.add_argument("--save_track", action='store_true', help="save the tracking video")
    parser.add_argument("--align", action='store_true', help="save the alignment results")
    parser.add_argument("--save_fps", default=15, help="the frame rate of the video")
    args = parser.parse_args()
    
    helper = Helper()
    if args.save_track:
        helper.make_video(args.track_path, args.track_path + '/tracking.mp4', args.save_fps)
    if args.align:
        helper.align_traj(args.root_path, args.traj_path)