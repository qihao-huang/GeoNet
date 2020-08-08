python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen_lab/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/convert_rgb_to_lab/model-50000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/convert_rgb_to_lab_test_lab"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/convert_rgb_to_lab_test_lab/model-50000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen_lab/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/convert_rgb_to_lab/model-100000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/convert_rgb_to_lab_test_lab"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/convert_rgb_to_lab_test_lab/model-100000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen_lab/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/convert_rgb_to_lab/model-150000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/convert_rgb_to_lab_test_lab"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/convert_rgb_to_lab_test_lab/model-150000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen_lab/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/convert_rgb_to_lab/model-200000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/convert_rgb_to_lab_test_lab"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/convert_rgb_to_lab_test_lab/model-200000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen_lab/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/convert_rgb_to_lab/model-250000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/convert_rgb_to_lab_test_lab" \
    --save_intermediate

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/convert_rgb_to_lab_test_lab/model-250000.npy"
