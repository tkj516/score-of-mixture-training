export CHECKPOINT_PATH=$1
export UPDATE_RATIO=${2:-5}

export WANDB_API_KEY=""
export WANDB_ENTITY=""
export WANDB_PROJECT="one-step-generation"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node 7 --nnodes 1 main/sjsd_dist/train_sjsd_dist.py \
    --generator_lr 5e-5  \
    --guidance_lr 5e-5  \
    --train_iters 400000 \
    --output_path  $CHECKPOINT_PATH/cifar10_sjsd_dist \
    --batch_size 40 \
    --initialie_generator --log_iters 500 \
    --resolution 32 \
    --label_dim 0 \
    --dataset_name "cifar10" \
    --seed 10 \
    --model_id pretrained_models/edm-cifar10-32x32-uncond-vp.pkl \
    --wandb_iters 150 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --use_fp16 \
    --wandb_name "cifar10_sjsd_dist"$UPDATE_RATIO   \
    --real_image_path datasets/cifar10 \
    --dfake_gen_update_ratio $UPDATE_RATIO \
    --delete_ckpts \
    --max_checkpoint 200 \
    --guidance_alpha_scheduler "augmented_discrete_uniform_sampler" \
    --generator_alpha_scheduler "partial_discrete_uniform_sampler" \
    --max_partitions 1001 \
    --point_mass_prob 0.25 \
    --wandb_hostname "" \
    --joint_training &  # Run the job in the background

# Wait for 5 minutes (600 seconds)
sleep 600

# Find the latest folder under the specified directory
LATEST_FOLDER=$(ls -d /home/tejasj/nobackup/alpha-skew-jsd/runs/cifar10_test/cifar10_sjsd_dist/*/ | sort -n | tail -n 1)

# Run the test command with the latest folder
CUDA_VISIBLE_DEVICES=7 python main/sjsd/test_folder_edm.py \
    --folder "$LATEST_FOLDER" \
    --wandb_name test_cifar10_sjsd_dist \
    --wandb_project one-step-generation \
    --resolution 32 \
    --label_dim 0 \
    --ref_path fid-refs/cifar10-32x32.npz \
    --detector_url pretrained_models/inception-2015-12-05.pkl \
    --dataset_name cifar10



