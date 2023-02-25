import torch
import torch.autograd as autograd
import torch.nn as nn
import copy
from utils import apply_mask, count_total_parameters, count_fc_parameters


def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y




def GraSP(diffusion, keep_ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=False):
    eps = 1e-10

    diffusion_copy = copy.deepcopy(diffusion).to(device)  # .eval()
    diffusion_copy.zero_grad()

    weights = []
    total_parameters = count_total_parameters(diffusion_copy)
    fc_parameters = count_fc_parameters(diffusion_copy)

    # rescale_weights(net)
    for layer in diffusion_copy.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # if isinstance(layer, nn.Linear) and reinit:
            #     nn.init.xavier_normal(layer.weight)
            weights.append(layer.weight)

    images_one = []
    targets_one = []

    grad_w = None
    for w in weights:
        w.requires_grad_(True)

    for it in range(num_iters):
        print("(1): Iterations %d/%d." % (it, num_iters))
        images, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
        N, c, h, w = images.shape
        din = copy.deepcopy(images)
        dtarget = copy.deepcopy(targets)
        images_one.append(din[:N//2])
        targets_one.append(dtarget[:N//2])
        images_one.append(din[N // 2:])
        targets_one.append(dtarget[N // 2:])

        images = images.to(device)
        targets = targets.to(device)

        t = torch.randint(0, diffusion.num_timesteps, (N,), device=device).long()
        loss = diffusion_copy.p_losses(images[:N//2], t=t[:N//2])
        grad_w_p = autograd.grad(loss, weights)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

        loss = diffusion_copy.p_losses(images[N // 2:], t=t[N // 2:])
        grad_w_p = autograd.grad(loss, weights, create_graph=False)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

    ret_inputs = []
    ret_targets = []

    for it in range(len(images_one)):
        print("(2): Iterations %d/%d." % (it, num_iters))
        images = images_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        ret_inputs.append(images)
        ret_targets.append(targets)

        t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device).long()
        loss = diffusion_copy.p_losses(images, t=t)

        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in diffusion_copy.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()
    grads = dict()
    model_modules = list(diffusion.modules())
    for idx, layer in enumerate(diffusion_copy.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads[model_modules[idx]] = -layer.weight.data * layer.weight.grad  # -theta_q Hg

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)
    diffusion.to(device)
    keep_masks = []
    for m, g in grads.items():
        keep_masks.append(((g / norm_factor) <= acceptable_score).float())
    apply_mask(diffusion, keep_masks)
