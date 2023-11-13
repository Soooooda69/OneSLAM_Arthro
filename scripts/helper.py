from natsort import natsorted
from tqdm import tqdm
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R

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

if __name__ == "__main__":
    helper = Helper()
    helper.make_video('../datasets/temp_data/localize_tracking','../datasets/temp_data/localize_tracking/tracking.mp4', 15)
