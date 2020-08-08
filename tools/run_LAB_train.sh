python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen_lab" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/convert_rgb_to_lab_2" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=300000 \
    --max_to_keep=40

exit