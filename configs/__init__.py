import argparse
import os

import numpy as np
import torch
import random


def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Non-deterministic")


def get_args():
    parser = argparse.ArgumentParser()
    # training specific args
    parser.add_argument('--dataset', type=str, default='voc12')
    parser.add_argument('--image_size', type=int, default=369)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrain', type=str, default=None)

    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--session_name', type=str)
    parser.add_argument('--fg_path', type=str, default='./datasets/voc12_iter0.npy')
    # models related params

    parser.add_argument('--num_epochs', type=int, default=20, help='This will affect learning rate decay')
    parser.add_argument('--stop_at_epoch', type=int, default=None)
    parser.add_argument('--start_at_epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=12)

    # optimization params
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='sgd, lars(from lars paper), lars_simclr(used in simclr and byol), larc(used in swav)')
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--final_lr', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5.0e-4)


    # infer params
    parser.add_argument('--cam_path', type=str)
    parser.add_argument('--crf_path', type=str)
    parser.add_argument('--adaptive_t', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0)

    #eval params
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--type', type=str)


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    return args
