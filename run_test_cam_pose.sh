# KITTI odometry dataset (including groundtruth poses)
# predicted pose snippets.

python geonet_main.py \
    --mode=test_pose \
    --dataset_dir="/userhome/34/h3567721/dataset/kitti/odometry/dataset/" \
    --init_ckpt_file="/userhome/34/h3567721/GeoNet/GeoNet_models_and_predictions/models/geonet_posenet" \
    --batch_size=1 \
    --seq_length=5 \
    --pose_test_seq=9 \
    --output_dir="/userhome/34/h3567721/projects/GeoNet/predictions/test_pose"

# generate the groundtruth pose snippets
python kitti_eval/generate_pose_snippets.py \
    --dataset_dir=/path/to/kitti/odom/dataset/ \
    --output_dir="/userhome/34/h3567721/dataset/kitti/GT_pose_snippets" \
    --seq_id=09 \
    --seq_length=5

# evaluate your predictions
python kitti_eval/eval_pose.py \
    --gtruth_dir="/userhome/34/h3567721/dataset/kitti/GT_pose_snippets" \
    --pred_dir="/userhome/34/h3567721/projects/GeoNet/predictions/eval_pose"