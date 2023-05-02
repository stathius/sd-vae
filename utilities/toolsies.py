import random
import numpy as np
import torch
import os
import re
import html
import urllib.parse
import argparse

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

def none_or_int(value):
    if value == 'None':
        return None
    return np.int(value)

def clean_filename(string: str) -> str:
    """
    Sanitize a string to be used as a filename.
    """
    string = html.unescape(string)
    string = urllib.parse.unquote(string)
    string = string.replace(':', '-').replace('/', '_').replace('\x00', '_')
    string = string.replace('}', '').replace('{', '').replace(',', '-')
    string = string.replace(' ', '').replace('\'', '')
    string = re.sub('[\n\\\*><?\"|\t]', '', string)
    string = string.strip()
    return string 

def run_cuda_diagnostics(requested_num_gpus):
    print("\nCUDA diagnostics:")
    print("-----------------")
    print("CUDA available? ", torch.cuda.is_available())
    print("Requested num devices: ", requested_num_gpus)
    print("Available num of devices: ", torch.cuda.device_count())
    print("CUDNN backend: ", torch.backends.cudnn.enabled)
    print("Distributed available: ", torch.distributed.is_available())
    print("NCCL available: ", torch.distributed.is_nccl_available())
    print("Distributed initialized: ", torch.distributed.is_initialized())
    assert requested_num_gpus <= torch.cuda.device_count(), "Not enough GPUs available."

def count_pars(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
