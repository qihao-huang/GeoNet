# depth and pose tasks 

export CUDA_VISIBLE_DEVICES=0

# train depth
python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir="/userhome/34/h3567721/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir="/userhome/34/h3567721/projects/GeoNet/checkpoint_depth" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=350000

# export CUDA_VISIBLE_DEVICES=1

# train pose
python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir="/userhome/34/h3567721/dataset/kitti/kitti_odom" \
    --checkpoint_dir="/userhome/34/h3567721/projects/GeoNet/checkpoint_pose" \
    --learning_rate=0.0002 \
    --seq_length=5 \
    --batch_size=4 \
    --max_steps=350000