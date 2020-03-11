# KITTI odometry dataset (including groundtruth poses)
PYTHON="/userhome/34/h3567721/anaconda3/envs/geonet-v/bin/python"

# predicted pose snippets.
$PYTHON geonet_main.py \
    --mode=test_pose \
    --dataset_dir=/path/to/kitti/odom/dataset/ \
    --init_ckpt_file=/path/to/trained/model/ \
    --batch_size=1 \
    --seq_length=5 \
    --pose_test_seq=9 \
    --output_dir=/path/to/save/predictions/

# generate the groundtruth pose snippets
$PYTHON kitti_eval/generate_pose_snippets.py \
    --dataset_dir=/path/to/kitti/odom/dataset/ \
    --output_dir=/path/to/save/gtruth/pose/snippets/ \
    --seq_id=09 \
    --seq_length=5

# evaluate your predictions
$PYTHON kitti_eval/eval_pose.py \
    --gtruth_dir=/path/to/gtruth/pose/snippets/ \
    --pred_dir=/path/to/predicted/pose/snippets/