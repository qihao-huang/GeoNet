# depth xyz
export CUDA_VISIBLE_DEVICES=0
user_path="/userhome/34/h3567721"

# train depth
python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=350000 \
    --log_savedir=$user_path"/projects/Depth/GeoNet/log/depth_geo_delta_two_stage" \
    --delta_mode \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/checkpoint/checkpoint_depth/model-240000"

    # two stage training strategy:
    #   1. tain the rigid to provide constraints/prior for the next stage
    #   2. load the model weight's without delta arch., then re-train the other stuff