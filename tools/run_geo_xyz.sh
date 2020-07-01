# depth xyz
export CUDA_VISIBLE_DEVICES=0
user_path="/userhome/34/h3567721"

# train depth
python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage_delta_init_0" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=1500001 \
    --log_savedir=$user_path"/projects/Depth/GeoNet/log/depth_geo_delta_two_stage_delta_init_0" \
    --delta_mode \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/checkpoint_depth/model-150000" \
    --max_to_keep=40 \
    --save_intermediate

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage_delta_init_0_fix_pose" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=1500001 \
    --log_savedir=$user_path"/projects/Depth/GeoNet/log/depth_geo_delta_two_stage_delta_init_0_fix_pose" \
    --delta_mode \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/checkpoint_depth/model-150000" \
    --fix_posenet \
    --max_to_keep=40 \
    --save_intermediate


python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage_4" \
    --learning_rate=0.0001 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=1500001 \
    --log_savedir=$user_path"/projects/Depth/GeoNet/log/depth_geo_delta_two_stage_4" \
    --delta_mode \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/checkpoint_depth/model-150000" \
    --max_to_keep=40 \
    --save_intermediate

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage_fix_pose_4" \
    --learning_rate=0.0001 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=1500001 \
    --log_savedir=$user_path"/projects/Depth/GeoNet/log/depth_geo_delta_two_stage_fix_pose_4" \
    --delta_mode \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/checkpoint_depth/model-150000" \
    --fix_posenet \
    --max_to_keep=40 \
    --save_intermediate

# ---------------------------------------------------------------------------------------------------------------------------

python geonet_main_semantic.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --semantic_dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen_seg/mask" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage_mask" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=350000 \
    --log_savedir=$user_path"/projects/Depth/GeoNet/log/depth_geo_delta_two_stage_mask" \
    --delta_mode \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/checkpoint/checkpoint_depth/model-240000"
