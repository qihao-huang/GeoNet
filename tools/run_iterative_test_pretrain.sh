
python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_fix_depth_1/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_depth_1"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_depth_1/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_fix_depth_3/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_depth_3"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_depth_3/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------


python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_fix_depth_5/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_depth_5"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_depth_5/model-10000.npy"


# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_fix_depth_7/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_depth_7"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_depth_7/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_fix_depth_9/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_depth_9"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_depth_9/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_fix_pose_2/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_pose_2"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_pose_2/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_fix_pose_4/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_pose_4"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_pose_4/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_fix_pose_6/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_pose_6"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_pose_6/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_fix_pose_8/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_pose_8"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_pose_8/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_fix_pose_10/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_pose_10"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/iterative_fix_pose_10/model-10000.npy"
