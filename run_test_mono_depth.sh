python geonet_main.py \
    --mode=test_depth \
    --dataset_dir="/userhome/34/h3567721/dataset/kitti/raw_data" \
    --init_ckpt_file="/userhome/34/h3567721/GeoNet/GeoNet_models_and_predictions/models/geonet_depthnet" \
    --batch_size=1 \
    --depth_test_split=eigen \
    --output_dir="/userhome/34/h3567721/projects/GeoNet/predictions/test_depth"

python kitti_eval/eval_depth.py \
    --split=eigen \
    --kitti_dir="/userhome/34/h3567721/dataset/kitti/raw_data" \
    --pred_file="/userhome/34/h3567721/projects/GeoNet/predictions/eval_depth"