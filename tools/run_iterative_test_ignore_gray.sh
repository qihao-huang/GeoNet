
# python geonet_main.py \
#     --mode=test_depth_vis \
#     --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
#     --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_iterative_fix_depth_1/model-10000" \
#     --batch_size=1 \
#     --depth_test_split=eigen \
#     --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_depth_1" \
#     --ignore_gray_warp

# python kitti_eval/eval_depth.py \
#     --split=eigen \
#     --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
#     --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_depth_1/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_iterative_fix_depth_3/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_depth_3" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_depth_3/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------


python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_iterative_fix_depth_5/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_depth_5" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_depth_5/model-10000.npy"


# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_iterative_fix_depth_7/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_depth_7" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_depth_7/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------


python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_iterative_fix_depth_9/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_depth_9" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_depth_9/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_iterative_fix_pose_2/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_pose_2" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_pose_2/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_iterative_fix_pose_4/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_pose_4" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_pose_4/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_iterative_fix_pose_6/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_pose_6" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_pose_6/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_iterative_fix_pose_8/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_pose_8" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_pose_8/model-10000.npy"

# --------------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_iterative_fix_pose_10/model-10000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_pose_10" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_iterative_fix_pose_10/model-10000.npy"
