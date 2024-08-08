import os

import numpy as np
import torch
import torch.utils.data as data
import soundfile as sf
from torchvision import transforms

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask,
                        random_bbox, random_cropping_bbox)

AUD_EXTENSIONS = ['.wav', '.midi']


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUD_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        audios = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        audios = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    audios.append(path)

    return audios


def soundfile_loader(f):
    segnp, fs = sf.read(f)
    return segnp


class InpaintDataset(data.Dataset):
    def __init__(self, data_root, sample_rate, mask_config={}, data_len=-1, audio_len=-1, loader=soundfile_loader,
                 sampled_bounds_path=None, skip_n_samples=-1):
        self.audios = make_dataset(data_root)

        if skip_n_samples > 0:
            self.audios = self.audios[skip_n_samples:]
        if data_len > 0:
            self.audios = self.audios[:int(data_len)]

        # self.tfs = transforms.Compose([
        #     transforms.Resize((image_size[0], image_size[1])),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # ])
        self.loader = loader
        self.audio_len = audio_len
        self.sample_rate = sample_rate
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']


    def __getitem__(self, index):
        ret = {}
        path = self.audios[index]
        audio = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_audio = audio * mask + mask * torch.randn_like(audio)
        mask_audio = audio * mask

        ret['gt_image'] = audio
        ret['cond_image'] = cond_audio
        ret['mask_image'] = mask_audio
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.audios)

    def tfs(self, audio):
        # Stereo to mono
        if len(audio) > 1:
            audio = np.mean(audio, axis=1)
            return audio

    def get_mask(self):
        mask = torch.ones((1, self.audio_len)).to(self.device)  # assume between 5 and 6s of total length
        if self.mask_mode == 'long':
            gap = int(self.mask_config[self.mask_mode]['gap_length'] * self.sample_rate / 1000)
            if self.mask_config[self.mask_mode]['start_gap_idx'] == "none":
                start_gap_index = int(self.audio_len // 2 - gap // 2)
            else:
                start_gap_index = int(self.mask_config[self.mask_mode]['start_gap_idx'] * self.sample_rate / 1000)
            mask[..., start_gap_index:(start_gap_index + gap)] = 0
        elif self.mask_mode == 'short':
            num_gaps = int(self.mask_config[self.mask_mode]['num_gaps'])
            gap_len = int(self.mask_config[self.mask_mode]['gap_length'] * self.sample_rate / 1000)
            if self.mask_config[self.mask_mode]['start_gap_idx'] == "none":
                start_gap_index = torch.randint(0, self.audio_len - gap_len, (num_gaps,))
                for i in range(num_gaps):
                    mask[..., start_gap_index[i]:(start_gap_index[i] + gap_len)] = 0
            else:
                start_gap_index = int(self.mask_config[self.mask_mode]['start_gap_idx'] * self.sample_rate / 1000)
            mask[..., start_gap_index:(start_gap_index + gap)] = 0
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return mask


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, sample_rate, mask_config={}, data_len=-1, audio_len=-1, loader=soundfile_loader):
        audios = make_dataset(data_root)
        if data_len > 0:
            self.audios = audios[:int(data_len)]
        else:
            self.audios = audios
        self.loader = loader
        self.audio_len = audio_len
        self.sample_rate = sample_rate
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']

    def __getitem__(self, index):
        ret = {}
        path = self.audios[index]
        audio = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_audio = audio * mask + mask * torch.randn_like(audio)
        mask_audio = audio * mask

        ret['gt_image'] = audio
        ret['cond_image'] = cond_audio
        ret['mask_image'] = mask_audio
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.audios)

    def tfs(self, audio):
        # Stereo to mono
        if len(audio) > 1:
            audio = np.mean(audio, axis=1)
            return audio

    def get_mask(self):
        mask = torch.ones((1, self.audio_len)).to(self.device)  # assume between 5 and 6s of total length
        if self.mask_mode == 'long':
            gap = int(self.mask_config[self.mask_mode]['gap_length'] * self.sample_rate / 1000)
            if self.mask_config[self.mask_mode]['start_gap_idx'] == "none":
                start_gap_index = int(self.audio_len // 2 - gap // 2)
            else:
                start_gap_index = int(self.mask_config[self.mask_mode]['start_gap_idx'] * self.sample_rate / 1000)
            mask[..., start_gap_index:(start_gap_index + gap)] = 0
        elif self.mask_mode == 'short':
            num_gaps = int(self.mask_config[self.mask_mode]['num_gaps'])
            gap_len = int(self.mask_config[self.mask_mode]['gap_length'] * self.sample_rate / 1000)
            if self.mask_config[self.mask_mode]['start_gap_idx'] == "none":
                start_gap_index = torch.randint(0, self.audio_len - gap_len, (num_gaps,))
                for i in range(num_gaps):
                    mask[..., start_gap_index[i]:(start_gap_index[i] + gap_len)] = 0
            else:
                start_gap_index = int(self.mask_config[self.mask_mode]['start_gap_idx'] * self.sample_rate / 1000)
            mask[..., start_gap_index:(start_gap_index + gap)] = 0
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return mask


class ColorizationDataset(data.Dataset):
    # This is a dummy dataset
    def __init__(self, data_root, sample_rate, data_flist, data_len=-1, audio_len=-1, loader=soundfile_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        # self.tfs = transforms.Compose([
        #     transforms.Resize((image_size[0], image_size[1])),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # ])
        self.loader = loader
        self.audio_len = audio_len
        self.sample_rate = sample_rate

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.wav'

        audio = self.tfs(self.loader(self.data_root))
        cond_image = self.tfs(self.loader(self.data_root))

        ret['gt_image'] = audio
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)

    def tfs(self, audio):
        # Stereo to mono
        if len(audio) > 1:
            audio = np.mean(audio, axis=1)
            return audio
