import wandb
import argparse
from glob import glob
from tqdm import tqdm
import shutil
import os


def main():
    parser = argparse.ArgumentParser('Clean logs')
    parser.add_argument('--username', type=str, default='pidan1231239', help='W&B username')
    parser.add_argument('--project', type=str, default='panomvsplat', help='W&B project')
    parser.add_argument('--logs_dir', type=str, default='logs', help='Logs directory')
    args = parser.parse_args()

    print('Retrieving list of W&B experiments')
    runs = wandb.Api().runs(f"{args.username}/{args.project}")
    exp_ids = [r.id for r in runs if 'archieved' not in r.tags]

    print('Scanning local experiments')
    exp_dirs = glob(os.path.join(args.logs_dir, '*/'))
    for exp_dir in tqdm(exp_dirs, desc='Cleaning'):
        if 'latest-run' in exp_dir:
            continue
        exp_id = os.path.basename(os.path.normpath(exp_dir))
        if exp_id not in exp_ids:
            tqdm.write(f"Deleting {exp_dir}")
            shutil.rmtree(exp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
