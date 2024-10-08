import logging
import os
from collections import OrderedDict

import cv2
import torch
import torch.nn as nn

from ..audio_network import extract
from .pinball import PinballLoss


class Conffusion(nn.Module):
    def __init__(self, baseModel, opt, device, load_finetuned=False):
        super(Conffusion, self).__init__()
        self.opt = opt
        self.device = device
        self.baseModel = baseModel.netG
        self.prediction_time_step = opt['train']['prediction_time_step']

        if opt['train']['finetune_loss'] == 'l2':
            self.criterion = nn.MSELoss(reduction='mean').to(self.device)
        elif opt['train']['finetune_loss'] == 'quantile_regression':
            self.q_lo_loss = PinballLoss(quantile=0.05)
            self.q_hi_loss = PinballLoss(quantile=0.95)
        self.log_dict = OrderedDict()
        self.lambda_hat = 0
        if load_finetuned:
            self.load_finetuned_network()

    def forward(self, masked_images, mask=None, gt_image=None, sigma=None):
        if 'audio' in self.baseModel.__name__.lower():
            masked_images = torch.squeeze(masked_images.type(torch.float32), 1)

        t = self.baseModel.diff_params.create_schedule(self.baseModel.num_timesteps).to(masked_images.device)

   
        sigma = t[0].unsqueeze(-1).to(masked_images.device)


        predicted_l,predicted_u = self.baseModel.diff_params.denoiser(masked_images, self.baseModel.denoise_fn,
                                                                       sigma.unsqueeze(-1),
                                                                       out_upper_lower=True)

        predicted_l.clamp_(-1., 1.)
        predicted_u.clamp_(-1., 1.)

        return predicted_l, predicted_u

    def bounds_regression_loss_fn(self, pred_l, pred_u, gt_l, gt_u):
        lower_loss = self.criterion(pred_l, gt_l)
        upper_loss = self.criterion(pred_u, gt_u)
        loss = lower_loss + upper_loss
        return loss

    def quantile_regression_loss_fn(self, pred_l, pred_u, gt_hr):
        # Step 1: Calculate the amplitude-based loss (quantile loss based on magnitudes)
        # Use absolute values for the amplitude comparison
        lower_loss_amplitude = self.q_lo_loss(torch.abs(pred_l), torch.abs(gt_hr))
        upper_loss_amplitude = self.q_hi_loss(torch.abs(pred_u), torch.abs(gt_hr))
        
        # Step 2: Calculate the phase-based loss (ensure the sign matches the ground truth)
        # Phase loss is the difference in signs between prediction and ground truth
        lower_loss_phase = torch.mean(torch.abs(torch.sign(pred_l) - torch.sign(gt_hr)))
        upper_loss_phase = torch.mean(torch.abs(torch.sign(pred_u) - torch.sign(gt_hr)))

        # Step 3: Combine amplitude and phase losses
        lower_loss = lower_loss_amplitude + lower_loss_phase
        upper_loss = upper_loss_amplitude + upper_loss_phase

        # Step 4: Total loss
        loss = lower_loss + upper_loss
        return loss

    def get_current_log(self):
        return self.log_dict


    def save_network(self, epoch, iter_step, optimizer, pred_lambda_hat, wandb_logger, network_name):
        """
        Save the network and optimizer state with a custom name.

        Args:
            epoch (int): Current epoch number.
            iter_step (int): Current iteration step.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            pred_lambda_hat (torch.Tensor): Model prediction.
            wandb_logger (wandb.Logger): Wandb logger for logging the model.
            network_name (str): The name of the network being saved (used for filenames).
        """
        
        # Define paths for generator (model) and optimizer checkpoints with a custom name
        gen_path = os.path.join(self.opt['path']['checkpoint'], f'{network_name}_gen.pth')
        opt_path = os.path.join(self.opt['path']['checkpoint'], f'{network_name}_opt.pth')
        
        # Save generator (model) state_dict
        state_dict = self.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        
        # Save optimizer state
        opt_state = {
            'epoch': epoch,
            'iter': iter_step,
            'pred_lambda_hat': pred_lambda_hat,
            'scheduler': None,  # Include scheduler state if applicable
            'optimizer': optimizer.state_dict()
        }
        torch.save(opt_state, opt_path)
        
        # Log to wandb
        wandb_logger.save(gen_path)
        wandb_logger.save(opt_path)

        # Optionally log additional info like metrics
        wandb_logger.log({"epoch": epoch, "iteration": iter_step, "model_saved_as": network_name})

    def save_best_network(self, epoch, iter_step, optimizer, pred_lambda_hat, wandb_logger):
        # Define paths for generator and optimizer checkpoints
        gen_path = os.path.join(self.opt['path']['checkpoint'], 'best_network_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(self.opt['path']['checkpoint'], 'best_network_opt.pth'.format(iter_step, epoch))
        
        # Save generator (model) state_dict
        state_dict = self.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        
        # Save optimizer state
        opt_state = {
            'epoch': epoch,
            'iter': iter_step,
            'pred_lambda_hat': pred_lambda_hat,
            'scheduler': None,
            'optimizer': optimizer.state_dict()
        }
        torch.save(opt_state, opt_path)
        
        # Log to wandb
        wandb_logger.save(gen_path)
        wandb_logger.save(opt_path)

        # Optionally log additional info like metrics
        wandb_logger.log({"epoch": epoch, "iteration": iter_step, "best_model_saved": True})


    def load_finetuned_network(self):
        load_path = self.opt['path']['bounds_resume_state']
        if load_path is not None:
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self
            if isinstance(self, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(gen_path), strict=(not self.opt['model']['finetune_norm']))
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

            saved_opt = torch.load(opt_path)
            self.pred_lambda_hat = saved_opt['pred_lambda_hat']
