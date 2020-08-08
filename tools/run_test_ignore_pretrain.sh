python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_warping_in_loss_pretrain/model-125000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain/model-125000.npy"

# --------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_warping_in_loss_pretrain/model-115000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain/model-115000.npy"

# --------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_warping_in_loss_pretrain/model-105000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain/model-105000.npy"

# --------------------------------------------------------------------------------------------------------


python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_warping_in_loss_pretrain/model-95000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain/model-95000.npy"

# --------------------------------------------------------------------------------------------------------


python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_warping_in_loss_pretrain/model-85000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain" \
    --ignore_gray_warp \
    --save_intermediate

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain/model-85000.npy"

# --------------------------------------------------------------------------------------------------------


python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_warping_in_loss_pretrain/model-75000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain/model-75000.npy"

# --------------------------------------------------------------------------------------------------------


python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_warping_in_loss_pretrain/model-65000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain/model-65000.npy"

# --------------------------------------------------------------------------------------------------------


python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_warping_in_loss_pretrain/model-55000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain/model-55000.npy"

# --------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_warping_in_loss_pretrain/model-45000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain/model-45000.npy"

# --------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_warping_in_loss_pretrain/model-35000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain/model-35000.npy"

# --------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_warping_in_loss_pretrain/model-25000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain/model-25000.npy"

# --------------------------------------------------------------------------------------------------------

python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/ignore_gray_warping_in_loss_pretrain/model-15000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain" \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/ignore_gray_warping_in_loss_pretrain/model-15000.npy"

# --------------------------------------------------------------------------------------------------------
