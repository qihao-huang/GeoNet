#!/bin/bash

# depth and pose tasks
# You can switch the network encoder by setting --dispnet_encoder flag, 
# or perform depth scale normalization (see this paper for details) by 
# setting --scale_normalize as True. 
# Note that for replicating depth and pose results, 
# the --seq_length is suggested to be 3 and 5 respectively.

PYTHON="/userhome/34/h3567721/anaconda3/envs/geonet-v/bin/python"

# train depth
CUDA_VISIBLE_DEVICES=0 $PYTHON geonet_main.py \
    --mode=train_rigid \
    --dataset_dir="/userhome/34/h3567721/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir="/userhome/34/h3567721/projects/GeoNet/checkpoint" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=350000
    # --dispnet_encoder \ 
    # --scale_normalize  

# train pose
# $PYTHON geonet_main.py \
#     --mode=train_rigid \
#     --dataset_dir="/userhome/34/h3567721/dataset/kitti/kitti_odom" \
#     --checkpoint_dir="/userhome/34/h3567721/projects/geonet/checkpoint" \
#     --learning_rate=0.0002 \
#     --seq_length=5 \
#     --batch_size=4 \
#     --max_steps=350000
#     # --dispnet_encoder \ 
#     # --scale_normalize  

# {
#  'add_dispnet': True,
#  'add_flownet': False,
#  'add_posenet': True,
#  'alpha_recon_image': 0.85,
#  'batch_size': 4,
#  'checkpoint_dir': '/userhome/34/h3567721/projects/geonet/checkpoint',
#  'dataset_dir': '/userhome/34/h3567721/dataset/kitti/kitti_raw_eigen',
#  'depth_test_split': 'eigen',
#  'disp_smooth_weight': 0.5,
#  'dispnet_encoder': 'resnet50',
#  'flow_consistency_alpha': 3.0,
#  'flow_consistency_beta': 0.05,
#  'flow_consistency_weight': 0.2,
#  'flow_smooth_weight': 0.2,
#  'flow_warp_weight': 1.0,
#  'flownet_type': 'residual',
#  'img_height': 128,
#  'img_width': 416,
#  'init_ckpt_file': None,
#  'learning_rate': 0.0002,
#  'max_steps': 350000,
#  'max_to_keep': 20,
#  'mode': 'train_rigid',
#  'num_scales': 4,
#  'num_source': 2,
#  'num_threads': 32,
#  'output_dir': None,
#  'pose_test_seq': 9,
#  'rigid_warp_weight': 1.0,
#  'save_ckpt_freq': 5000,
#  'scale_normalize': False,
#  'seq_length': 3
# }