# Filename: infer_musicldm.py
'''
This code is released and maintained by:

Ke Chen, Yusong Wu, Haohe Liu
MusicLDM: Enhancing Novelty in Text-to-Music Generation Using Beat-Synchronous Mixup Strategies
All rights reserved

contact: knutchen@ucsd.edu
'''
import sys

sys.path.append("src")

import os

import numpy as np

import argparse
import yaml
import torch

from pytorch_lightning.strategies.ddp import DDPStrategy
from src.latent_diffusion.models.musicldm import MusicLDM
from src.utilities.data.dataset import TextDataset

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from dataloader import MusicDataset, get_dataloader

config_path = 'musicldm.yaml'


def main(config, seed):

    seed_everything(seed)
    os.makedirs(config['cache_location'], exist_ok=True)
    os.environ['TRANSFORMERS_CACHE'] = config['cache_location']
    torch.hub.set_dir(config['cache_location'])
    os.makedirs(config['log_directory'], exist_ok=True)
    log_path = os.path.join(config['log_directory'], os.getlogin())
    os.makedirs(log_path, exist_ok=True)
    folder_name = os.listdir(log_path)
    i = 0
    while str(i) in folder_name:
        i = i + 1
    log_path = os.path.join(log_path, str(i))
    os.makedirs(log_path, exist_ok=True)

    print(f'Log files will be saved at {log_path}')

    batch_size = config["model"]["params"]["batchsize"]

    # Preparing data for the model
    train_dataset = MusicDataset('training_data')  # replace 'training_data' with your directory
    valid_dataset = MusicDataset('validation_data')  # replace 'validation_data' with your directory

    train_loader = get_dataloader('train', batch_size=batch_size, shuffle=True)
    valid_loader = get_dataloader('validation', batch_size=batch_size, shuffle=False)

    # Training configuration
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=log_path,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min',
    )

    devices = torch.cuda.device_count()

    latent_diffusion = MusicLDM(**config["model"]["params"])
    latent_diffusion.set_log_dir(log_path, log_path, log_path)
    trainer = Trainer(
        max_epochs=config['model']['params']['epochs'],
        accelerator="gpu",
        callback=[checkpoint_callback],
        devices=devices,
        strategy=DDPStrategy(find_unused_parameters=False)
        if (int(devices) > 1)
        else None,
    )

    # The training step
    trainer.fit(latent_diffusion, train_loader, valid_loader)

    print(f"Training Finished. Please check the log files and the saved models at {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="a generation seed",
        default=0
    )

    args = parser.parse_args()

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    main(config, args.seed)

