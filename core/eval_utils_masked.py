import torch

from .calibration_masked import fraction_missed_loss


def get_rcps_metrics(lower_bound, upper_bound, ground_truth, masks):
  risks_losses = []
  sizes = []
  for idx in range(upper_bound.shape[0]):
    risks_losses.append(fraction_missed_loss(lower_bound[idx].unsqueeze(0), upper_bound[idx].unsqueeze(0), ground_truth[idx].unsqueeze(0),masks[idx].unsqueeze(0),only_avg_masked=True, avg_channels=False).item())
    mask_indices = torch.argwhere(masks[idx] == 1.)
    sizes.append((upper_bound[idx] - lower_bound[idx])[mask_indices[:, 0]].mean().unsqueeze(dim=0).item())


  # risks_losses = torch.stack(risks_losses, dim=0)
  # sizes = torch.stack(sizes, dim=0)
  risks_losses = torch.tensor(risks_losses).to(lower_bound.device)
  sizes =torch.tensor(sizes).to(lower_bound.device)
  sizes = sizes + torch.rand(size=sizes.shape,device=sizes.device) * 1e-6

  size_bins = torch.tensor([0, torch.quantile(sizes, 0.25), torch.quantile(sizes, 0.5), torch.quantile(sizes, 0.75)], device=sizes.device)
  buckets = torch.bucketize(sizes, size_bins)-1
  stratified_risks = torch.tensor([torch.nan_to_num(risks_losses[buckets == bucket].mean()) for bucket in range(size_bins.shape[0])])

  return risks_losses.mean(), sizes.mean(), sizes.median(), stratified_risks