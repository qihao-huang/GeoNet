export CUDA_VISIBLE_DEVICES=0

user_path="/userhome/34/h3567721"


# NOTE: pay attention to test_depth_delta
python geonet_main.py \
    --mode=test_depth_delta \
    --dataset_dir=$user_path"/dataset/kitti/raw_data/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta/model-250000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet/predictions/test_xyz_depth" \
    --delta_mode

# python kitti_eval/eval_depth.py \
#     --split=eigen \
#     --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
#     --pred_file=$user_path"/projects/Depth/GeoNet/predictions/test_depth/model-240000.npy"