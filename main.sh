# --pretrained-ckpt 'trained_models/Unet_cifar10-epoch_130-timesteps_1000-class_condn_False.pt'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --diffusion-steps 1000 --sampling-steps 250 --density 1.0 --epochs 500 --batch-size 128
