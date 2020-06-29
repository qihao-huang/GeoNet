# depth and pose tasks 

export CUDA_VISIBLE_DEVICES=0

user_path="/userhome/34/h3567721"

# train depth, seq_length=3
python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint/checkpoint_depth_3" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=350001 \
    --log_savedir=$user_path"/projects/Depth/GeoNet/log/depth_3" \
    --max_to_keep=80

# train pose, seq_length=5
# python geonet_main.py \
#     --mode=train_rigid \
#     --dataset_dir=$user_path"/dataset/kitti/kitti_odom" \
#     --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint/checkpoint_pose" \
#     --learning_rate=0.0002 \
#     --seq_length=5 \
#     --batch_size=4 \
#     --max_steps=350000