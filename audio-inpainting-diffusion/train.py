import os
import re
import json
import hydra
#import click
import torch
import utils.dnnlib as dnnlib
from utils.torch_utils import distributed as dist
import utils.setup as setup
from training.trainer import Trainer

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.


def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------


def _main(args):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #assert torch.cuda.is_available()
    #device="cuda"

    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if dist.get_rank() == 0:
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
    args.exp.model_dir=args.model_dir


    #opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')

    #dist.init()
    dset=setup.setup_dataset(args)
    diff_params=setup.setup_diff_parameters(args)
    network=setup.setup_network(args, device)
    optimizer=setup.setup_optimizer(args, network)
    #try:
    test_set=setup.setup_dataset_test(args)
    #except:
    #    test_set=None
    tester=setup.setup_tester(args, network=network, diff_params=diff_params, test_set=test_set, device=device) #this will be used for making demos during training
    trainer=setup.setup_trainer(args, dset=dset, network=network, optimizer=optimizer, diff_params=diff_params, tester=tester, device=device) #this will be used for making demos during training


    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0()
    dist.print0(f'Output directory:        {args.model_dir}')
    dist.print0(f'Network architecture:    {args.network.callable}')
    dist.print0(f'Diffusion parameterization:  {args.diff_params.callable}')
    dist.print0(f'Batch size:              {args.exp.batch}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Mixed-precision:         {args.exp.use_fp16}')
    dist.print0()

    # Train.
    #trainer=Trainer(args=args, dset=dset, network=network, optimizer=optimizer, diff_params=diff_params, tester=tester, device=device)
    trainer.training_loop()

@hydra.main(config_path="conf", config_name="conf")
def main(args):
    _main(args)

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
