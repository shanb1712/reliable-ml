from core.praser import mkdirs
import hydra
import pandas as pd
import shutil
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.


def _main(args):
    # Define groups for splitting
    split_groups = ['validation', 'calibration', 'test']

    # Define paths for splits
    split_path = {}
    for split in split_groups:
        split_path[split] = f"{args.dset.path}splits/{split}/ground_truth"
        mkdirs(split_path[split])  # Ensure the directory exists

    # Load metadata
    metadata_file = os.path.join(args.dset.path, "maestro-v3.0.0.csv")
    metadata = pd.read_csv(metadata_file)
    metadata = metadata[metadata["year"].isin(args.dset.years+args.dset.years_test+args.dset.years_calibration)]
    metadata.loc[metadata.year.isin(args.dset.years_calibration) & metadata.split.isin(
        ['validation', 'test']), 'split'] = 'calibration'

    # Create file list for each group
    filelist = {}
    for group in split_groups:
        filelist[group] = metadata.groupby('split').get_group(group)['audio_filename']
        filelist[group] = filelist[group].map(lambda x: os.path.join(args.dset.path, x), na_action='ignore').to_list()

    # Move files to the appropriate split directories
    for group, files in filelist.items():
        for file in files:
            if os.path.isfile(file):
                shutil.move(file, split_path[group])
        print(f'{group} contains {len(files)} audio files,')
        print(f'{len(next(os.walk(split_path[group]))[-1])} successfully moved to {split_path[group]}.')


@hydra.main(config_path="config", config_name="conf")
def main(args):
    _main(args)


if __name__ == "__main__":
    main()