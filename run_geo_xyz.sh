# depth xyz

export CUDA_VISIBLE_DEVICES=0

user_path="/userhome/34/h3567721"

# train depth
python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint_depth" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=350000