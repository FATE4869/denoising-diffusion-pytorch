import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

def Random_pruning(diffusion, keep_ratio, device):
    for layer in diffusion.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            prune.random_unstructured(layer, name='weight', amount=1-keep_ratio)

    #
    # for i, (name, module) in enumerate(modules.named_modules()):
    #     # for name, param in model.named_parameters():
    #     if isinstance(module, unets.TimestepEmbedSequential):
    #         for inner_name, inner_module in module.named_modules():
    #             # sum += inner_module.numel()
    #             if isinstance(inner_module, torch.nn.Conv2d):
    #                 prune.random_unstructured(inner_module, name='weight', amount=pr)
    #                 prune.random_unstructured(inner_module, name='bias', amount=pr)
    #                 # print(f'{count}\t {name}: {inner_name}\t{inner_module} with number of weights: {torch.numel(inner_module.weight)}')
    #                 # print(f'{count}\t {name}: {inner_name}\t{inner_module} with number of bias: {torch.numel(inner_module.bias)}')
    #                 # count += 2
    #             elif isinstance(inner_module, torch.nn.Linear):
    #                 prune.random_unstructured(inner_module, name='weight', amount=pr)
    #                 prune.random_unstructured(inner_module, name='bias', amount=pr)
    #                 # print(f'{count}\t {name}: {inner_name}\t{inner_module} with number of weights: {torch.numel(inner_module.weight)}')
    #                 # print(f'{count}\t {name}: {inner_name}\t{inner_module} with number of bias: {torch.numel(inner_module.bias)}')
    #                 # count += 2
    #             elif isinstance(inner_module, unets.GroupNorm32):
    #                 prune.random_unstructured(inner_module, name='weight', amount=pr)
    #                 prune.random_unstructured(inner_module, name='bias', amount=pr)
    #                 # print(f'{count}\t {name}: {inner_name}\t{inner_module} with number of weights: {torch.numel(inner_module.weight)}')
    #                 # print(f'{count}\t {name}: {inner_name}\t{inner_module} with number of bias: {torch.numel(inner_module.bias)}')
    #                 # count += 2