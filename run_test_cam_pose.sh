# KITTI odometry dataset (including groundtruth poses)
# predicted pose snippets.

export CUDA_VISIBLE_DEVICES=0

user_path="/userhome/34/h3567721"

python geonet_main.py \
    --mode=test_pose \
    --dataset_dir=$user_path"/dataset/kitti/odometry/dataset/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/checkpoint/checkpoint_pose/model-345000" \
    --batch_size=1 \
    --seq_length=5 \
    --pose_test_seq=9 \
    --output_dir=$user_path"/projects/Depth/GeoNet/predictions/test_pose/"

# generate the groundtruth pose snippets
python kitti_eval/generate_pose_snippets.py \
    --dataset_dir=$user_path"/dataset/kitti/odometry/dataset/" \
    --output_dir=$user_path"/dataset/kitti/GT_pose_snippets/" \
    --seq_id=09 \
    --seq_length=5

# evaluate your predictions
python kitti_eval/eval_pose.py \
    --gtruth_dir=$user_path"/dataset/kitti/GT_pose_snippets/" \
    --pred_dir=$user_path"/projects/Depth/GeoNet/predictions/test_pose/"

# ATE mean: 0.0134, std: 0.0079