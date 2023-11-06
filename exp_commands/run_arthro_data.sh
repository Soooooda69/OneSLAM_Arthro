#!/bin/bash

# data_root="./data/ARTHROSCOPY/h_1_500"
# data_root0="../../datasets/e"
# data_root1="../../datasets/g"
data_root="../../datasets/h"
experiment_folder="./experiments/Arthroscopy"

python run_slam.py --data_root $data_root --output_folder $experiment_folder --name h_1_400 --process_subset --start_idx 1 --end_idx 200 \
                    --section_length 13 --image_subsample 1 --point_sampler r2d2 --tracked_point_num_min 200 \
                    --cotracker_model cotracker_stride_4_wind_8 --update_localized_pose --ransac_localization --gt_pose_path $data_root/poses_gt.txt \
                    --verbose # --pose_guesser gt_pose --no_localize

#data_root="./data/ARTHROSCOPY/g_800_1300"
#python run_slam.py --data_root $data_root --output_folder $experiment_folder --name g_800_1300 --process_subset --start_idx 800 --end_idx 1300 --image_subsample 1 --point_sampler r2d2 --tracked_point_num_min 200 --cotracker_model cotracker_stride_4_wind_8

#data_root="./data/ARTHROSCOPY/e"
#python run_slam.py --data_root $data_root --output_folder $experiment_folder --name e --point_sampler r2d2 --tracked_point_num_min 200 --cotracker_model cotracker_stride_4_wind_8

#!/bin/bash


