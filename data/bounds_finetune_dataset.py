import os

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import soundfile as sf

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask,
                        random_bbox, random_cropping_bbox)
import utils.training_utils as t_utils

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

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


def make_dataset_extracted_bounds(dir):
    if os.path.isfile(dir):
        audios = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        audios = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                audios.append(path)

    return audios

def soundfile_loader(f):
    segnp, fs = sf.read(f)
    return segnp, fs


def resample_audio(audio, fs, sample_rate, audio_len):
    # this has been reused from the trainer.py
    return t_utils.resample_batch(audio, fs, sample_rate, audio_len)


class BoundsInpaintDataset(data.Dataset):
    def __init__(self, data_root, sample_rate, mask_config={}, load_len=0, data_len=-1, audio_len=-1,
                 loader=soundfile_loader,
                 sampled_bounds_path=None, skip_n_samples=-1):
        self.loader = loader
        self.audio_len = audio_len  # Final length of audio after resampling
        self.sample_rate = sample_rate
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']

        self.data_len = data_len  # how many audio to load
        self.load_len = load_len  # length of segment from the audio
        self.masked_length = self.mask_config[self.mask_mode]['gap_length']

        self.audios = make_dataset(data_root)
        lower_bounds = make_dataset_extracted_bounds(f"{sampled_bounds_path}/{self.masked_length}/lower_bounds")
        upper_bounds = make_dataset_extracted_bounds(f"{sampled_bounds_path}/{self.masked_length}/upper_bounds")
        sampled_masks = make_dataset_extracted_bounds(f"{sampled_bounds_path}/{self.masked_length}/masks")
        masked_samples = make_dataset_extracted_bounds(f"{sampled_bounds_path}/{self.masked_length}/masked_samples")

        if self.data_len > 0:
            self.audios = self.audios[:int(data_len)]
            self.lower_bounds = lower_bounds[:int(data_len)]
            self.upper_bounds = upper_bounds[:int(data_len)]
            self.sampled_masks = sampled_masks[:int(data_len)]
            self.masked_samples = masked_samples[:int(data_len)]
        else:
            self.audios = self.audios[:int(data_len)]
            self.lower_bounds = lower_bounds
            self.upper_bounds = upper_bounds
            self.sampled_masks = sampled_masks
            self.masked_samples = masked_samples

        self.load_dataset()

    def __getitem__(self, index):
        ret = {}
        original = torch.from_numpy(self.test_samples[index])[None, :]
        mask = self.get_mask()

        seg = resample_audio(original, torch.from_numpy(np.array(self.f_s[index])), self.sample_rate, self.audio_len)
        mask_audio = seg * mask

        cond_audio = seg * mask + mask * torch.randn_like(seg)

        lower_bound = torch.load(self.lower_bounds[index])
        upper_bound = torch.load(self.upper_bounds[index])

        masked_samples = torch.load(self.masked_samples[index]).squeeze(dim=0)
        sampled_masks = torch.load(self.sampled_masks[index]).squeeze(dim=1)

        ret['lower_bound'] = lower_bound
        ret['upper_bound'] = upper_bound
        ret['masked_samples'] = masked_samples
        ret['sampled_masks'] = sampled_masks

        ret['gt_image'] = seg
        ret['cond_image'] = cond_audio
        ret['mask_image'] = mask_audio
        ret['mask'] = mask
        ret['path'] = self.audios[index].rsplit("/")[-1].rsplit("\\")[-1]

        return ret

    def __len__(self):
        return len(self.test_samples)

    def tfs(self, audio):
        # Stereo to mono
        if len(audio) > 1:
            audio = np.mean(audio, axis=1)
            return audio

    def load_dataset(self):
        # TODO: load all dataset: upper, lower, sampled, masked
        self.test_samples = []
        self.filenames = []
        self.f_s = []
        for i in range(len(self.audios)):
            file = self.audios[i]
            self.filenames.append(os.path.basename(file))
            data, samplerate = self.loader(file)
            data = self.tfs(data)

            self.test_samples.append(data[10 * samplerate:10 * samplerate + self.load_len])  # use only 50s
            self.f_s.append(samplerate)
        return

    def get_mask(self):
        mask = np.ones((1, self.audio_len))  # assume between 5 and 6s of total length
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
        return torch.from_numpy(mask)  # 1, L


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
        img = self.tfs(self.loader(path))
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

    def get_mask(self):
        mask = torch.ones((1, self.audio_len))  # assume between 5 and 6s of total length
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
    def __init__(self, data_root, sample_rate, data_flist, data_len=-1, audio_len=-1, loader=soundfile_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        # self.tfs = transforms.Compose([
        #         transforms.Resize((image_size[0], image_size[1])),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
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
