import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.utils.prune as prune
import time
# Preliminaries. Not to be exported.

def _is_prunable_module(m):
    return (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d))


# get the sparsity
def _get_sparsity(tsr):
    total = tsr.numel()
    nnz = tsr.nonzero().size(0)
    return nnz / total


# get the number of non-zero elements
def _get_nnz(tsr):
    return tsr.nonzero().size(0)

def count_total_parameters(model):
    total = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel()
    return total


def count_fc_parameters(model):
    total = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear)):
            total += m.weight.numel()
    return total

# Modules

def get_weights(model):
    weights = []
    for m in model.modules():
        if _is_prunable_module(m):
            weights.append(m.weight)
    return weights


def get_convweights(model):
    weights = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weights.append(m.weight)
    return weights


def get_modules(model):
    modules = []
    for m in model.modules():
        if _is_prunable_module(m):
            modules.append(m)
    return modules


def get_convmodules(model):
    modules = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            modules.append(m)
    return modules


def get_copied_modules(model):
    modules = []
    for m in model.modules():
        if _is_prunable_module(m):
            modules.append(deepcopy(m).cpu())
    return modules


def get_model_sparsity(model, verbose=False):
    prunables = 0
    nnzs = 0
    for i, (name, m) in enumerate(model.named_modules()):
        if _is_prunable_module(m):
            if verbose:
                print(f'{name} \t {m}: num of nonzeros / num of total parameters: {m.weight.data.nonzero().size(0)} / {m.weight.data.numel()}')
            prunables += m.weight.data.numel()
            nnzs += m.weight.data.nonzero().size(0)

    return nnzs / prunables


def get_sparsities(model):
    return [_get_sparsity(m.weight.data) for m in model.modules() if _is_prunable_module(m)]


def get_nnzs(model):
    return [_get_nnz(m.weight.data) for m in model.modules() if _is_prunable_module(m)]

def apply_mask(model, masks):
    module_list = get_modules(model)
    for i, m in enumerate(module_list):
        prune.custom_from_mask(m, name='weight', mask=masks[i])

class loss_logger:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.loss = []
        self.start_time = time.time()
        self.ema_loss = None
        self.ema_w = 0.9

    def log(self, v, display=False):
        self.loss.append(v)
        if self.ema_loss is None:
            self.ema_loss = v
        else:
            self.ema_loss = self.ema_w * self.ema_loss + (1 - self.ema_w) * v

        if display:
            print(
                f"Steps: {len(self.loss)}/{self.max_steps} \t loss (ema): {self.ema_loss:.3f} "
                + f"\t Time elapsed: {(time.time() - self.start_time)/3600:.3f} hr"
            )