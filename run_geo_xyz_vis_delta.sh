# depth xyz
conda_env=$(echo $CONDA_DEFAULT_ENV)

if [ $conda_env == "geonet-v" ]
then
    echo $conda_env
    echo "Running with geonet-v, Python 2.7, Tensorflow 1.14"
else
    echo $conda_env
    conda activate geonet-v  
    echo "Setting geonet-v, Python 2.7, Tensorflow 1.14"
fi

export CUDA_VISIBLE_DEVICES=0

user_path="/userhome/34/h3567721"

# $user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta/model-250000"

# train depth
python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_vis" \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet/checkpoint/depth_geo_delta_two_stage/model-100000" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=350000 \
    --log_savedir=$user_path"/projects/Depth/GeoNet/log/depth_geo_delta_vis_two_stage" \
    --delta_mode \
    --save_intermedia