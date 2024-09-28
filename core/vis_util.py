import numpy as np
import core.util as Util
import matplotlib.pyplot as plt
import os

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

def cheack_overlapping(wandb_logger, pred_l, pred_u, masked_input, mask_audio):
    # Convert tensors to NumPy arrays (assuming they are 1D vectors for audio)
    pred_lower_bound_audio = pred_l.detach().cpu().numpy().squeeze()
    pred_upper_bound_audio = pred_u.detach().cpu().numpy().squeeze()
    masked_sample_audio = masked_input.detach().cpu().numpy().squeeze()
    mask_audio = mask_audio.numpy().squeeze()
    ci_intervals = (pred_upper_bound_audio-pred_lower_bound_audio)
    gap_length = len(mask_audio[0][mask_audio[0]==0])
    overlap_percentage_list = []
    image_series = []
    filenames = []
    for i in range(pred_lower_bound_audio.shape[0]):
        # Sum of the first 100 elements
        cumsum_vector = np.cumsum(ci_intervals[i])
        window_sums = cumsum_vector[gap_length:] - cumsum_vector[:-gap_length]
        max_index = np.argmax(window_sums)
        
        #Create the mask for the higest CI vec (in length of the gap)
        highlight_mask = np.zeros_like(ci_intervals[i])
        highlight_mask[max_index:max_index + gap_length] = 1  # Mark the 100-length highest area

        # find start and end idx of mask
        min_mask = np.where(mask_audio[i]==0)[0].min()
        max_mask = np.where(mask_audio[i]==0)[0].max()
        
        # Calculate the overlapping region
        overlap_min = max(max_index, min_mask)
        overlap_max = min(max_index+gap_length, max_mask)
        if overlap_min < overlap_max:  # There is an overlap
            overlap_percentage = (overlap_max - overlap_min)/gap_length*100
        else:
            overlap_percentage = 0
        overlap_percentage_list.append(overlap_percentage)
        
        save_dir = 'figures_overlap'
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))

        # Plot the original vector
        plt.plot(ci_intervals[i], label="CI", alpha=0.6)
        # plt.plot(pred_lower_bound_audio, label="pred_l", alpha=0.6)
        # plt.plot(pred_upper_bound_audio, label="pred_u", alpha=0.6)
        # plt.plot(masked_sample_audio, label="masked_input", alpha=0.6)

        # Shade the area for the highest 100 consecutive values
        plt.axvspan(max_index, max_index + gap_length, color='red', alpha=0.3, label="Higest CI")

        # Shade the area for the custom-defined mask
        plt.axvspan(min_mask, max_mask, color='blue', alpha=0.3, label="Mask")

        # Add labels and legend
        plt.legend()
        plt.title(f"overlap = {overlap_percentage:.2f}%")
        plt.xlabel("t")
        plt.ylabel("CI")

        # Save the plot to a file and log it to wandb
        plt.savefig(f"{save_dir}/test_audio_{i}.png")
        plt.close()

        img = plt.imread(f"{save_dir}/test_audio_{i}.png")

        # Add the image to the series with an optional caption
        image_series.append(img)
        filenames.append(f'Image Series/test_audio_{i}')
        # Concatenate all audio clips into one array
        # Prepare audio clips and corresponding captions

    return image_series, filenames, np.mean(overlap_percentage_list), np.median(overlap_percentage_list)

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