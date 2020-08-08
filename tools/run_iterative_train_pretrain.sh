python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_1" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/checkpoint_depth/model-150000" \
    --max_to_keep=4 \
    --fix_depth

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_pose_2" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_1/model-10000" \
    --max_to_keep=4 \
    --fix_pose

python geonet_main.py\
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_3" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_pose_2/model-10000" \ 
    --max_to_keep=4 \
    --fix_depth

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_pose_4" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_3/model-10000" \ 
    --max_to_keep=4 \
    --fix_pose

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_5" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_pose_4/model-10000" \ 
    --max_to_keep=4 \
    --fix_depth

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_pose_6" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_5/model-10000" \ 
    --max_to_keep=4 \
    --fix_pose

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_7" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_pose_6/model-10000" \ 
    --max_to_keep=4 \
    --fix_depth

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_pose_8" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_7/model-10000" \ 
    --max_to_keep=4 \
    --fix_pose

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_9" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_pose_8/model-10000" \ 
    --max_to_keep=4 \
    --fix_depth

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_pose_10" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_9/model-10000" \ 
    --max_to_keep=4 \
    --fix_pose

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_11" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_pose_10/model-10000" \ 
    --max_to_keep=4 \
    --fix_depth

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_pose_12" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_11/model-10000" \ 
    --max_to_keep=4 \
    --fix_pose

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_13" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_pose_12/model-10000" \ 
    --max_to_keep=4 \
    --fix_depth

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_pose_14" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_13/model-10000" \ 
    --max_to_keep=4 \
    --fix_pose

python geonet_main.py \
    --mode=train_rigid \
    --dataset_dir=$user_path"/dataset/kitti/kitti_raw_eigen" \
    --checkpoint_dir=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_depth_15" \
    --learning_rate=0.0002 \
    --seq_length=3 \
    --batch_size=4 \
    --max_steps=10001 \
    --init_ckpt_file=$user_path"/projects/Depth/GeoNet-ori/checkpoint/iterative_scratch_fix_pose_14/model-10000" \ 
    --max_to_keep=4 \
    --fix_depth