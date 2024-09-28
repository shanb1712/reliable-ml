# Con*ffusion*-Audio: Confidence Intervals for audio-inpainting Diffusion Models

## Overview

In this project, we extend the Conffusion approach to the audio domain, specifically targeting classical music inpainting. We fine-tune a pre-trained audio inpainting model to denoise noisy audio data and predict upper and lower bounds for each reconstructed segment in a single pass. Through experiments, we explore whether these interval predictions can help identify artificially inpainted sections in audio, improving the detection of synthesized or modified segments in audio restoration tasks.

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
wget -P checkpoints/maestro/ https://huggingface.co/Eloimoliner/audio-inpainting-diffusion/resolve/main/maestro_22k_8s-750000.pt
```
Update the `resume_state` under `path` in the `extract_bounds_inpainting_center_confussion.json` to the path of the pretrained model `maestro_22k_8s-750000.pt`.
## Dataset
### Download data
Download the MAESTRO Dataset V3.0.0 and move the dataset to the `dldata/` directory
```angular2html
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
```
### Split data
Split the data to calibration, validation and test sets by running the `create_split.py` script.
```angular2html
python3 create_split.py
```

## Pre-train model
### Extract boundaries
Run the `extract_bounds.py` script for each data split
```angular2html
python3 extract_bounds.py -p {calibration/test/validation} -c config/extract_bounds_inpainting_center_conffusion.json --distributed_worker_id 0
```
This will create new audio samples with generated areas of length `gap_length` as defined in `config/extract_bounds_inpainting_center_conffusion.json`, extract lower and upper bound for those and masked areas.
For the rest of this project, we only tested bounds by Conffusion, thus the we will only used the new generated audio created by the script.

### Finetune model
Run the `inpainting_finetune_bounds.py` script for each `gap_length`
```angular2html
python3 inpainting_finetune_bounds.py
```
Update the `resume_state` under `path` in the `finetune_bounds_inpainting_center_nconffusion.json` to the path of the pretrained model `maestro_22k_8s-750000.pt`
for finetuning.
The fine-tuned model will be saved under `experiment/<wandb-project-name>/checkpoint`.

### Test fine-tuned model
Run the `inpainting_finetune_bounds.py` script for each `gap_length`
```angular2html
python3 test_finetuned_bounds.py
```
Update the `resume_state` under `path` in the `test_finetune_bounds_inpainting_center_nconffusion.json` to the path of the finetuned model under `experiment/<wandb-project-name>/checkpoint/best_model`
and `gap_length` to test the desired length of generated data to test.

## Results

## Citation

- **Horwitz, Eliahu and Hoshen, Yedid. (2022).** Conffusion: Confidence Intervals for Diffusion Models. *arXiv preprint arXiv:2211.09795*. Available at: [https://arxiv.org/abs/2211.09795](https://arxiv.org/abs/2211.09795)
  - GitHub Repository: [Link](https://github.com/eliahuhorwitz/Conffusion)
- **Moliner Juanpere, Eloi, and Vesa Välimäki. (2024).** "Diffusion-Based Audio Inpainting." *Audio Engineering Society*. Available at: [https://aaltodoc.aalto.fi/items/37bc136c-6818-4190-a650-f3fca61c4df9](https://aaltodoc.aalto.fi/items/37bc136c-6818-4190-a650-f3fca61c4df9)
  - GitHub Repository: [Link](https://github.com/eloimoliner/audio-inpainting-diffusion)
- **Hawthorne, Curtis, et al. (2019).** Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset. *Proceedings of the International Conference on Learning Representations (ICLR)*. Available at: [https://openreview.net/forum?id=r1lYRjC9F7](https://openreview.net/forum?id=r1lYRjC9F7)
