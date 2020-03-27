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

# train depth
python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet/checkpoint_depth_test" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=350000 \
    --log_savedir=$user_path"/projects/Depth/GeoNet/log"
