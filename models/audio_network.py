import math
from functools import partial
from inspect import isfunction

import numpy as np
import torch
from core.base_network import BaseNetwork
from tqdm import tqdm
import hydra
import utils.setup as setup
import copy
from datetime import date
import utils.dnnlib as dnnlib
import utils.training_utils as t_utils
import utils.logging as utils_logging

import os

from .cqtdiff_maestro.unet_cqt_oct_with_projattention_adaLN_2 import Unet_CQT_oct_with_attention as UNet


class Audio_Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='cqtdiff_maestro', **kwargs):
        super(Audio_Network, self).__init__(**kwargs)

        with hydra.initialize(config_path="../config"):
            args = hydra.compose(config_name="conf.yaml", return_hydra_config=True)
        self.args = args
        self.do_inpainting = 'inpainting' in self.args.tester.modes
        self.paths = {}
        self.setup_paths()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.denoise_fn = setup.setup_network(args, self.device)
        self.network = self.denoise_fn
        self.diff_params = setup.setup_diff_parameters(args)
        # self.beta_schedule = beta_schedule
        torch.backends.cudnn.benchmark = True
        self.setup_sampler()
        self.use_wandb = False
        today = date.today()

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = self.diff_params.create_schedule(self.args.tester.T)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        gammas = self.diff_params.get_gamma(betas).detach().cpu().numpy()
        gammas_prev = np.append(1., gammas[:-1])

        betas = betas.detach().cpu().numpy()
        alphas = 1. - betas

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
                extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(y_t, t=t,
                                                noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
                sample_gammas.sqrt() * y_0 +
                (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    # @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        if not (self.rid):
            pred = self.sampler.predict_inpainting(y_t, mask)
        else:
            pred, rid_denoised, rid_grads, rid_grad_update, rid_pocs, rid_xt, rid_xt2, t = self.sampler.predict_inpainting(
                mask, y_t)

        return pred, []

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        if mask is not None:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy * mask + (1. - mask) * y_0], dim=1), sample_gammas)
            loss = self.loss_fn(mask * noise, mask * noise_hat)
        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
            loss = self.loss_fn(noise, noise_hat)
        return loss

    """
    # ------------------------------------------------------------------------------- #
    # -------------------- Functions relative to this network only ------------------ #
    # ------------------------------------------------------------------------------- #
    """

    def setup_sampler(self):
        self.rid = False
        self.sampler = dnnlib.call_func_by_name(func_name=self.args.tester.sampler_callable, model=self.network,
                                                diff_params=self.diff_params, args=self.args, rid=self.rid)

    def load_checkpoint(self, path, map_location):
        state_dict = torch.load(path, map_location=map_location)
        if self.network.finetune_bounds:
            from collections import OrderedDict
            new_state_dict_network = OrderedDict()
            new_state_dict_ema = OrderedDict()
            for k, v in state_dict['network'].items():
                if 'ups.' in k:
                    name_upper = k.replace("ups", "ups_upper")
                    name_lower = k.replace("ups", "ups_lower")
                    new_state_dict_network[name_upper] = v
                    new_state_dict_network[name_lower] = v
                else:
                    new_state_dict_network[k] = v
            for k, v in state_dict['ema'].items():
                if 'ups.' in k:
                    name_upper = k.replace("ups", "ups_upper")
                    name_lower = k.replace("ups", "ups_lower")
                    new_state_dict_ema[name_upper] = v
                    new_state_dict_ema[name_lower] = v
                else:
                    new_state_dict_ema[k] = v
            state_dict.pop('network')
            state_dict.pop('ema')
            state_dict['network'] = new_state_dict_network
            state_dict['ema'] = new_state_dict_ema
        try:
            self.it = state_dict['it']
        except:
            self.it = 0
        print("loading checkpoint")
        return t_utils.load_state_dict(state_dict, ema=self.network)

    def setup_paths(self):
        if self.do_inpainting and ("inpainting" in self.args.tester.modes):
            mode = "inpainting"
            self.paths[mode], self.paths[mode + "degraded"], self.paths[mode + "original"], self.paths[
                mode + "reconstructed"] = self.prepare_experiment("inpainting", "masked", "inpainted")

    def prepare_experiment(self, str, str_degraded="degraded", str_reconstruced="recosntucted"):
        path_exp = os.path.join(self.args.model_dir, str)
        if not os.path.exists(path_exp):
            os.makedirs(path_exp)

        n = str_degraded
        path_degraded = os.path.join(path_exp, n)  # path for the lowpassed
        # ensure the path exists
        if not os.path.exists(path_degraded):
            os.makedirs(path_degraded)

        path_original = os.path.join(path_exp, "original")  # this will need a better organization
        # ensure the path exists
        if not os.path.exists(path_original):
            os.makedirs(path_original)

        n = str_reconstruced
        path_reconstructed = os.path.join(path_exp, n)  # path for the clipped outputs
        # ensure the path exists
        if not os.path.exists(path_reconstructed):
            os.makedirs(path_reconstructed)

        return path_exp, path_degraded, path_original, path_reconstructed

# gaussian diffusion trainer class
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas