import torch
import argparse
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, train_one_epoch
from torch.utils.data import DataLoader
from data import get_metadata, get_dataset, fix_legacy_dict
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import loss_logger
import torch.nn.utils.prune as prune

import os
import cv2
import copy
import time
# import math
# from tqdm import tqdm
# from easydict import EasyDict
# import torch.distributed as dist

# import torch.nn.utils.prune as prune
# import pickle
import pruner.snip as snip
import pruner.grasp as grasp
import pruner.random_pruning as random_pruning
import utils

def main(args):
    args.device = "cuda:{}".format(args.local_rank)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)

    model = Unet(dim=64)
    diffusion = GaussianDiffusion(model, image_size=32, timesteps=1000, loss_type='l2', sampling_timesteps=250).to(args.device)
    if args.local_rank == 0:
        print(f'The total number of parameter: {utils.count_total_parameters(diffusion)}')
        print(f'The sparsity is:{utils.get_model_sparsity(diffusion)}')
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)

    metadata = get_metadata(args.dataset)
    train_set = get_dataset(name=args.dataset, data_dir=args.data_dir, metadata=metadata)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=2,
                              pin_memory=True)  # train_loader for pruning only

    if args.density < 1.0:
        # pruners
        random_pruning.Random_pruning(diffusion=diffusion, keep_ratio=args.density, device=args.device)
        # snip.SNIP(diffusion=diffusion, keep_ratio=args.density, train_dataloader=train_loader, device=args.device)
        # grasp.GraSP(diffusion=diffusion, keep_ratio=args.density, train_dataloader=train_loader, device=args.device,
        #             num_classes=10, samples_per_class=20)
        print(f'The sparsity after pruning is:{utils.get_model_sparsity(diffusion)}')
    if args.local_rank == 0:
        print(f"Training dataset loaded: Number of batches: {len(train_loader)}, Number of images: {len(train_set)}")
    return 0
    current_epoch = 0
    if args.pretrained_ckpt:
        print(f"Loading pretrained model from {args.pretrained_ckpt}")
        d = fix_legacy_dict(torch.load(args.pretrained_ckpt, map_location=args.device))
        dm = diffusion.state_dict()
        if args.delete_keys:
            for k in args.delete_keys:
                print(
                    f"Deleting key {k} becuase its shape in ckpt ({d[k].shape}) doesn't match "
                    + f"with shape in model ({dm[k].shape})"
                )
                del d[k]

        current_epoch = d['epoch']
        diffusion.load_state_dict(d['model'], strict=False)
        optimizer.load_state_dict(d['opt'])
        # args.ema_dict = copy.deepcopy(diffusion.state_dict())
        args.ema_dict = d['ema']
        print(
            f"Mismatched keys in ckpt and model: ",
            set(d['model'].keys()) ^ set(dm.keys()),
        )
        print(f"Loaded pretrained model from {args.pretrained_ckpt}")
        print(f"Density is: {utils.get_model_sparsity(model, verbose=False)}")


    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        if args.local_rank == 0:
            print(f"Using distributed training on {ngpus} gpus.")
        args.batch_size = args.batch_size // ngpus
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        diffusion = DDP(diffusion, device_ids=[args.local_rank], output_device=args.local_rank)
        sampler = DistributedSampler(train_set)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    logger = loss_logger(len(train_loader) * args.epochs)

    # ema model
    if not args.pretrained_ckpt:
        args.ema_dict = copy.deepcopy(diffusion.state_dict())

    if args.local_rank == 0:
        result = {
            'epoch': current_epoch,
            'model': diffusion.module.state_dict(),
            'opt': optimizer.state_dict(),
            'ema': args.ema_dict,
        }
        torch.save(result, os.path.join(args.save_dir,
                                        f"{args.arch}_{args.dataset}-epoch_{current_epoch}-timesteps_"
                                        f"{args.diffusion_steps}-class_condn_{args.class_cond}.pt"))

    # lets start training the model
    for epoch in range(args.epochs):
        start = time.time()
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_one_epoch(diffusion, train_loader, optimizer, logger, None, args)
        current_epoch += 1
        if (epoch + 1) % 10 == 0:
            if args.local_rank == 0:
                result = {
                    'epoch': current_epoch,
                    'model': diffusion.module.state_dict(),
                    'opt': optimizer.state_dict(),
                    'ema': args.ema_dict,
                }
                torch.save(result, os.path.join(args.save_dir, f"{args.arch}_{args.dataset}-epoch_{current_epoch}-timesteps_"
                                   f"{args.diffusion_steps}-class_condn_{args.class_cond}.pt"))
        end = time.time()
        print(f'{epoch+1} / {args.epochs}\t time: {(end - start)/60}min')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Minimal implementation of diffusion models")
    # diffusion model
    parser.add_argument("--arch", type=str,default='Unet', help="Neural network architecture")
    parser.add_argument(
        "--class-cond",
        action="store_true",
        default=False,
        help="train class-conditioned diffusion model",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=1000,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=250,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        default=False,
        help="Sampling using DDIM update step",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.2,
        help="density of weights in the model",
    )
    # dataset
    parser.add_argument("--dataset", type=str, default='cifar10')
    parser.add_argument("--data-dir", type=str, default="./dataset/")
    # optimizer
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch-size per gpu"
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--ema_w", type=float, default=0.9995)
    # sampling/finetuning
    parser.add_argument("--pretrained-ckpt", type=str, help="Pretrained model ckpt")
    parser.add_argument("--delete-keys", nargs="+", help="Pretrained model ckpt")
    parser.add_argument(
        "--sampling-only",
        action="store_true",
        default=False,
        help="No training, just sample images (will save them in --save-dir)",
    )
    parser.add_argument(
        "--num-sampled-images",
        type=int,
        default=50000,
        help="Number of images required to sample from the model",
    )

    # misc
    parser.add_argument("--save-dir", type=str, default="./trained_models/")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--seed", default=112233, type=int)
    args = parser.parse_args()

    main(args)