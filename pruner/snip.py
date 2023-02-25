import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_weights, get_modules, get_model_sparsity, apply_mask
import copy
import types
from torch.nn.utils import prune

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP(diffusion, keep_ratio, train_dataloader, device):
    # TODO: shuffle?
    diffusion.to(device)
    # Grab a single batch from the training dataset
    images, targets = next(iter(train_dataloader))
    b, c, h, w = images.shape
    images = diffusion.normalize(images).to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    diffusion_copy = copy.deepcopy(diffusion).to(device)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in diffusion_copy.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            # nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    diffusion_copy.zero_grad()
    t = torch.randint(0, diffusion.num_timesteps, (b,), device=device).long()

    loss = diffusion_copy.p_losses(images, t=t)
    # xt, eps = diffusion.sample_from_forward_process(inputs, time_steps)
    # loss = ((pred_eps - eps) ** 2).mean()
    loss.backward()

    grads_abs = []
    for layer in diffusion_copy.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    apply_mask(diffusion.model, keep_masks)
    # diffusion.to('cpu')
