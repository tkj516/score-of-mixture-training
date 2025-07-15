export CHECKPOINT_PATH=$1
export UPDATE_RATIO=${2:-5}

export WANDB_API_KEY=""
export WANDB_ENTITY=""
export WANDB_PROJECT="one-step-generation"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node 7 --nnodes 1 main/sjsd/train_sjsd.py \
    --generator_lr 5e-6  \
    --guidance_lr 5e-5  \
    --train_iters 400000 \
    --output_path  $CHECKPOINT_PATH/imagenet_sjsd \
    --batch_size 40 \
    --log_iters 500 \
    --resolution 64 \
    --label_dim 1000 \
    --dataset_name "imagenet" \
    --seed 10 \
    --model_id pretrained_models/edm-imagenet-64x64-cond-adm.pkl \
    --wandb_iters 150 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --use_fp16 \
    --wandb_name "imagenet_sjsd"$UPDATE_RATIO   \
    --real_image_path datasets/imagenet/imagenet-64x64_lmdb \
    --dfake_gen_update_ratio $UPDATE_RATIO \
    --gan_classifier \
    --cls_loss_weight 1e-2 \
    --gen_cls_loss_weight 3e-3 \
    --diffusion_gan \
    --diffusion_gan_max_timestep 1000 \
    --delete_ckpts \
    --max_checkpoint 200 \
    --guidance_alpha_scheduler "augmented_discrete_uniform_sampler" \
    --generator_alpha_scheduler "zeroless_discrete_uniform_sampler" \
    --max_partitions 1001 \
    --fake_score_arch "unet" \
    --wandb_hostname "" \
    --multistep_generator \
    --num_generator_steps 1 \
    --lr_scheduler "cosine" \
    --cache_dir $CHECKPOINT_PATH/imagenet_sjsd_cache \
    --max_checkpoint 200 &  # Run the job in the background

# Wait for 5 minutes (300 seconds)
sleep 300

# Find the latest folder under the specified directory
LATEST_FOLDER=$(ls -d runs/imagenet_test/imagenet_sjsd/*/ | sort -n | tail -n 1)

# Run the test command with the latest folder
CUDA_VISIBLE_DEVICES=7 python main/sjsd/test_folder_edm.py \
    --folder "$LATEST_FOLDER" \
    --wandb_name test_imagenet_sjsd \
    --wandb_project one-step-generation \
    --resolution 64 \
    --label_dim 1000 \
    --ref_path fid-refs/imagenet_fid_refs_edm.npz \
    --detector_url pretrained_models/inception-2015-12-05.pkl \
    --dataset_name imagenet

