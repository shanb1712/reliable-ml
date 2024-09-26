import numpy as np
import core.util as Util


def create_image_grid(pred_l, pred_u, partial_gt, masked_input, gt_sample, n_rows=5):
    try:
        image_rows = []
        for i in range(n_rows):
            image_row = create_image_row(pred_l=pred_l[i].detach(), pred_u=pred_u[i].detach(), partial_gt=partial_gt[i],
                                         masked_input=masked_input[i], gt_sample=gt_sample[i])
            image_rows.append(image_row)
        image_grid = np.concatenate(image_rows, axis=0)
        return image_grid
    except:
        print("Error while creating image grid!")
        return None

def create_image_row(pred_l, pred_u, partial_gt, masked_input, gt_sample):
    pred_lower_bound_img = Util.tensor2img(pred_l)
    pred_upper_bound_img = Util.tensor2img(pred_u)
    masked_sample_img = Util.tensor2img(masked_input)
    gt_sample_img = Util.tensor2img(gt_sample)
    partial_gt_img = Util.tensor2img(partial_gt)
    image_row = np.concatenate((pred_lower_bound_img, pred_upper_bound_img, partial_gt_img, masked_sample_img, gt_sample_img), axis=1)
    return image_row

def create_audio_row(pred_l, pred_u, masked_input, gt_sample):
    # Convert tensors to NumPy arrays (assuming they are 1D vectors for audio)
    pred_lower_bound_audio = pred_l.detach().cpu().numpy().squeeze()
    pred_upper_bound_audio = pred_u.detach().cpu().numpy().squeeze()
    masked_sample_audio = masked_input.detach().cpu().numpy().squeeze()
    gt_sample_audio = gt_sample.detach().cpu().numpy().squeeze()
    ci_intervals = (pred_upper_bound_audio-pred_lower_bound_audio)


    # Concatenate all audio clips into one array
    # Prepare audio clips and corresponding captions
    audio_clips = [
        (pred_lower_bound_audio, "Pred Lower Bound"),
        (pred_upper_bound_audio, "Pred Upper Bound"),
        (masked_sample_audio, "Masked Input"),
        (gt_sample_audio, "Full Ground Truth"),
        (ci_intervals, "CI intervals (upper-lower)")
    ]

    return audio_clips

def create_audio_grid(pred_l, pred_u, masked_input, gt_sample):
    audio_rows = []
    
    for i in range(len(pred_l)):
        # Create a row of audio clips
        audio_row = create_audio_row(pred_l[i], pred_u[i], masked_input[i], gt_sample[i])
        audio_rows.append(audio_row)

    return audio_rows

def log_audio_grid_as_list(wandb_logger, audio_rows, sample_rate=22050):
    try:
        audio_log = []
        captions = []
        
        # Prepare a list of audio clips and their captions
        for idx, audio_row in enumerate(audio_rows):
            for audio_clip, caption in audio_row:
                audio_log.append(wandb_logger.Audio(audio_clip, sample_rate=sample_rate, caption=f"Row {idx+1}: " + caption))
                
        # Log all the audio clips as a single grouped entry under "Validation/Audio"
        wandb_logger.log({
            "Validation/Audio": audio_log
        })
    except Exception as e:
        print(f"Error while logging audio grid: {e}")

def log_train(diffusion_with_bounds, wandb_logger, pred_l, pred_u, cond_image, gt_image):
    logs = diffusion_with_bounds.get_current_log()
    audio_clips = create_audio_row(pred_l=pred_l[0].detach().cpu(), pred_u=pred_u[0].detach().cpu(),
                                    masked_input=cond_image[0].detach().cpu(), gt_sample=gt_image[0].detach().cpu())

    if wandb_logger:
        # wandb_logger.log_image("Finetune/Images", image_to_log, caption="Pred L, Pred U, GT, Masked Input, Full GT", commit=False)
        # Create a list to log all audio clips in one call
        audio_logs = {f"Finetune/Audio {caption}": wandb_logger._wandb.Audio(audio_clip, sample_rate=22050, caption=caption)
                      for audio_clip, caption in audio_clips}

        # Log all audio clips together (they will be shown together in the wandb dashboard)
        wandb_logger._wandb.log(audio_logs)

        wandb_logger.log_metrics(logs)