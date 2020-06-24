export CUDA_VISIBLE_DEVICES=0

user_path="/userhome/34/h3567721"

python geonet_main.py \
    --mode=test_depth \
    --dataset_dir=$user_path"/dataset/kitti/raw_data/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/checkpoint/checkpoint_depth/model-240000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet/predictions/test_depth"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet/predictions/test_depth/model-240000.npy"