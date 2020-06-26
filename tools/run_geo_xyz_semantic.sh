# depth xyz
export CUDA_VISIBLE_DEVICES=0
user_path="/userhome/34/h3567721"

# train depth
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

python geonet_main_semantic.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --semantic_dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen_seg/mask" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage_mask_fix_pose" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=350000 \
    --log_savedir=$user_path"/projects/Depth/GeoNet/log/depth_geo_delta_two_stage_mask_fix_pose" \
    --delta_mode \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/checkpoint/checkpoint_depth/model-240000" \
    --fix_posenet