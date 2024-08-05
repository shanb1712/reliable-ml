# Con*ffusion*-Audio: Confidence Intervals for audio-inpainting Diffusion Models

## Overview

Brief description of your project, its purpose, and how it can be used.

## Prerequisites

Ensure you have the following installed:
- Python 3.8.19
- CUDA=11.8
- Installation of the required library dependencies:
```angular2html
python3 -m venv conffusion_venv
source conffusion_venv/bin/activate
pip install -r requirements.txt
```

## Pretrained model
Download musicnet pre-trained model from hugging face
```angular2html
wget -P experiments/cqtdiff+_MUSICNET https://huggingface.co/Eloimoliner/audio-inpainting-diffusion/resolve/main/musicnet_44k_4s-560000.pt
```

## Dataset
Download the MAESTRO Dataset V3.0.0
```angular2html
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
```

## Citation

- **Horwitz, Eliahu and Hoshen, Yedid. (2022).** Conffusion: Confidence Intervals for Diffusion Models. *arXiv preprint arXiv:2211.09795*. Available at: [https://arxiv.org/abs/2211.09795](https://arxiv.org/abs/2211.09795)
  - GitHub Repository: [Link](https://github.com/eliahuhorwitz/Conffusion)
- **Moliner Juanpere, Eloi, and Vesa Välimäki. (2024).** "Diffusion-Based Audio Inpainting." *Audio Engineering Society*. Available at: [https://aaltodoc.aalto.fi/items/37bc136c-6818-4190-a650-f3fca61c4df9](https://aaltodoc.aalto.fi/items/37bc136c-6818-4190-a650-f3fca61c4df9)
  - GitHub Repository: [Link](https://github.com/eloimoliner/audio-inpainting-diffusion)
- **Hawthorne, Curtis, et al. (2019).** Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset. *Proceedings of the International Conference on Learning Representations (ICLR)*. Available at: [https://openreview.net/forum?id=r1lYRjC9F7](https://openreview.net/forum?id=r1lYRjC9F7)
