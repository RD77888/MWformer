import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends

from exp.exp import Exp_main
from utils.print_args import print_args

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def main():

    set_seed(42)
    EXPERIMENT_MAP = {
        'Exp': Exp_main,
    }
    parser = argparse.ArgumentParser(description='Runoff Forecasting Workflow')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--task_id', type=str, required=True, default='test', help='task id')
    parser.add_argument('--model', type=str, required=True, default='MWformer',
                        help='model name')

    # Data Loader
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='data.csv', help='data file')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau', help='set scheduler type, options: [reduce_on_plateau, onecycle]')
    # Forecasting Task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=24, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

    # Model Define
    parser.add_argument('--num_runoff', type=int, default=6, help='runoff input size')
    parser.add_argument('--num_rain', type=int, default=6, help='rain input size')
    parser.add_argument('--patch_num', type=int, default=24, help='time period number')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--wavelet_name', type=str, default='db2', help='wavelet type for wavelet transform')
    parser.add_argument('--level', type=int, default=2, help='level for multi-level wavelet decomposition')


    parser.add_argument('--MTL_loss', type=str, default='SMOOTH_L1', help='loss name, options: [L1, L2, SMOOTH_L1]')
    parser.add_argument('--loss', type=str, default='MTL', help='train use criterion, options: [L1, L2, MTL]')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp_main', choices=EXPERIMENT_MAP.keys(),
                        help='Which experiment class to use')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    # --- FIXED: Restored your original, more robust device setup logic ---
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'args_setting')
    os.makedirs(save_dir, exist_ok=True)

    json_file_name = f"{args.model}_args.json"
    json_file_path = os.path.join(save_dir, json_file_name)
    with open(json_file_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print(f'Using GPU: cuda:{args.gpu}')
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
            print('Using Apple Metal (MPS)')
        else:
            args.device = torch.device("cpu")
            print('Using CPU')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        print(f'Using Multi-GPU: {args.device_ids}, primary on cuda:{args.gpu}')

    print('Args in experiment:')
    print_args(args)

    Exp = EXPERIMENT_MAP.get(args.des)
    if Exp is None:
        raise ValueError(f"Experiment type '{args.des}' not found!")

    # --- Workflow Execution ---
    for ii in range(args.itr):
        print(f"\n>>>>>>> Starting Iteration: {ii + 1}/{args.itr} <<<<<<<")
        seed = 42
        set_seed(seed)
        exp = Exp(args)
        setting = f'{args.model}_taskid{args.task_id}_sl{args.seq_len}_pl{args.pred_len}_itr{ii}_{args.loss}'
        exp.train(setting)
        exp.test(setting,test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
