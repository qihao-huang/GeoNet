# depth xyz
export CUDA_VISIBLE_DEVICES=0

user_path="/userhome/34/h3567721"

# to save intermedia variables to visulize 
python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage_vis_tmp" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage/model-100000" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=350000 \
    --log_savedir=$user_path"/projects/Depth/GeoNet/log/depth_geo_delta_two_stage_vis_tmp" \
    --delta_mode \
    --save_intermedia
    
python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage_fix_pose_2_vis" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage_fix_pose_2/model-100000" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=350001 \
    --log_savedir=$user_path"/projects/Depth/GeoNet/log/depth_geo_delta_two_stage_fix_pose_2_vis" \
    --delta_mode \
    --save_intermediate

# ---------------------------------------------------------------------------------------------------------------------------

python geonet_main_semantic.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --semantic_dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen_seg/mask" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage_mask_fix_pose_vis" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage_mask_fix_pose/model-65000" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=350000 \
    --log_savedir=$user_path"/projects/Depth/GeoNet/log/depth_geo_delta_two_stage_mask_fix_pose_vis_2" \
    --delta_mode \
    --save_intermediate \
    --fix_posenet