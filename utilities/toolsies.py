import random
import numpy as np
import torch
import os
import argparse
import sys

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def none_or_float(value):
    if value == 'None':
        return None
    return np.float(value)

def run_cuda_diagnostics(requested_num_gpus):
    print("\nCUDA diagnostics:")
    print("-----------------")
    print("CUDA available? ", torch.cuda.is_available())
    print("Requested num devices: ", requested_num_gpus)
    print("Available num of devices: ", torch.cuda.device_count())
    print("CUDNN backend: ", torch.backends.cudnn.enabled)
    assert requested_num_gpus <= torch.cuda.device_count(), "Not enough GPUs available."

def error_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    print(*args, file=sys.stdout, **kwargs)

def terminate_on_nan(loss):
    if torch.isnan(loss).any():
        error_print("Terminating program -- NaN detected.")
        exit()

def count_pars(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def list2string(list_to_parse):
    output = ""
    for list_elem in list_to_parse:
        output += str(list_elem) + "_"
    return output