export CUDA_VISIBLE_DEVICES=0

user_path="/userhome/34/h3567721"

python geonet_main.py \
    --mode=test_depth_delta \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_sigmoid_minus_0.5_scale_5_semantic/model-150000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet/predictions/depth_geo_delta_sigmoid_minus_0.5_scale_5_semantic" \
    --delta_mode \
    --save_intermediate

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet/predictions/depth_geo_delta_sigmoid_minus_0.5_scale_5/model-150000.npy"

python geonet_main.py \å
    --mode=test_depth_delta \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_sigmoid_minus_0.5_scale_5/model-200000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet/predictions/depth_geo_delta_sigmoid_minus_0.5_scale_5" \
    --delta_mode \
    --save_intermediate

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet/predictions/depth_geo_delta_sigmoid_minus_0.5_scale_5/model-200000.npy"
    