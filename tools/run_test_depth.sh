python geonet_main.py \
    --mode=test_depth_vis \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/checkpoint_depth/model-150000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/checkpoint_depth_tmp" \
    --save_intermediate \
    --ignore_gray_warp

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet-ori/predictions/checkpoint_depth/model-150000.npy"

# ----------------------------------------------------

exp_name="checkpoint_depth"

python test_metric_script.py \
    --mode=test_depth \
    --dataset_dir=$user_path"/dataset/kitti/raw_data/" \
    --ckpt_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/"$exp_name \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/test_"$exp_name \
    --seq_length=3
