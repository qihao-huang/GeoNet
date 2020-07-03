export CUDA_VISIBLE_DEVICES=0

user_path="/userhome/34/h3567721"

# NOTE: pay attention to test_depth_delta
python geonet_main.py \
    --mode=test_depth_delta \
    --dataset_dir=$user_path"/dataset/kitti/kitti_depth_test_eigen/" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage_delta_sigmoid/model-140000" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet/predictions/depth_geo_delta_two_stage_delta_sigmoid" \
    --delta_mode \
    --save_intermediate

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --pred_file=$user_path"/projects/Depth/GeoNet/predictions/depth_geo_delta_two_stage_delta_init_0_lr2/model-50000.npy"

# ---------------------------------------------------------------------------------------------------------------------------
exp_name="checkpoint_depth"

python test_metric_script.py \
    --mode=test_depth_delta \
    --dataset_dir=$user_path"/dataset/kitti/raw_data/" \
    --ckpt_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/"$exp_name \
    --kitti_dir=$user_path"/dataset/kitti/raw_data/" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir=$user_path"/projects/Depth/GeoNet-ori/predictions/test_"$exp_name \
    --delta_mode \
    --seq_length=3
