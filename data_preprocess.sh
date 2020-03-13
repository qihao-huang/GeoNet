# Python 2.7, TensorFlow 1.1 and CUDA 8.0 on Ubuntu 16.04.

PYTHON="/userhome/34/h3567721/anaconda3/envs/geonet-v/bin/python"

# preprocessing
# depth
# For depth task, the --dataset_name should be kitti_raw_eigen and --seq_length is set to 3;
python data/prepare_train_data.py \
    --dataset_dir="/userhome/34/h3567721/dataset/kitti/raw_data/" \
    --dataset_name=kitti_raw_eigen \
    --dump_root="/userhome/34/h3567721/dataset/kitti/kitti_raw_eigen/" \
    --seq_length=3 \
    --img_height=128 \
    --img_width=416 \
    --num_threads=16 \
    --remove_static

# flow
# For flow task, the --dataset_name should be kitti_raw_stereo and --seq_length is set to 3;
python data/prepare_train_data.py \
    --dataset_dir="/userhome/34/h3567721/dataset/kitti/raw_data/" \
    --dataset_name=kitti_raw_stereo \
    --dump_root="/userhome/34/h3567721/dataset/kitti/kitti_raw_stereo/" \
    --seq_length=3 \
    --img_height=128 \
    --img_width=416 \
    --num_threads=16 \
    --remove_static

# pose
# For pose task, the --dataset_name should be kitti_odom and --seq_length is set to 5.
python data/prepare_train_data.py \
    --dataset_dir="/userhome/34/h3567721/dataset/kitti/odometry/dataset/" \
    --dataset_name=kitti_odom \
    --dump_root="/userhome/34/h3567721/dataset/kitti/kitti_odom/" \
    --seq_length=5 \
    --img_height=128 \
    --img_width=416 \
    --num_threads=16 \
    --remove_static