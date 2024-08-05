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