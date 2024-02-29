#!/bin/bash

# data_root="./data/ARTHROSCOPY/h_1_500"
# data_root0="../../datasets/e"
# data_root1="../../datasets/g"
data_root="../datasets/h"
experiment_folder="./experiments"

rm -r ../datasets/temp_data/localize_tracking/*

# # h_good_1
python slam.py --data_root $data_root --output_folder $experiment_folder --name test_h --process_subset --start_idx 1 --end_idx 200 \
                    --section_length 4 --past_frame_size 3 --local_ba_size 12 --image_subsample 1 --point_sampler r2d2 --tracked_point_num_min 600 \
                    --cotracker_model cotracker_stride_4_wind_8 --update_localized_pose --gt_pose_path $data_root/poses_gt.txt \
                    --depth_network_model vitl --tracking_ba_iterations 10\
                    #--cotracker_model cotracker2 cotracker_stride_4_wind_8 #--no_localize #--dense_ba #--verbose # --pose_guesser gt_pose --no_localize

#g
# python slam.py --data_root $data_root --output_folder $experiment_folder --name test_g --process_subset --start_idx 1 --end_idx 300 \
#                     --section_length 4 --past_frame_size 3 --local_ba_size 12 --image_subsample 1 --point_sampler r2d2 --tracked_point_num_min 400 \
#                     --cotracker_model cotracker_stride_4_wind_8 --update_localized_pose --ransac_localization --gt_pose_path $data_root/poses_gt.txt \
#                     --depth_network_model vitl \
#                     #--cotracker_model cotracker2 cotracker_stride_4_wind_8 #--no_localize #--dense_ba #--verbose # --pose_guesser gt_pose --no_localize

# #e
# python slam.py --data_root $data_root --output_folder $experiment_folder --name test_e --process_subset --start_idx 1 --end_idx 300 \
#                     --section_length 4 --past_frame_size 3 --local_ba_size 12 --image_subsample 1 --point_sampler r2d2 --tracked_point_num_min 400 \
#                     --cotracker_model cotracker_stride_4_wind_8 --update_localized_pose --ransac_localization --gt_pose_path $data_root/poses_gt.txt \
#                     --depth_network_model vitl \
#                     #--cotracker_model cotracker2 cotracker_stride_4_wind_8 #--no_localize #--dense_ba #--verbose # --pose_guesser gt_pose --no_localize

python ./scripts/helper.py






