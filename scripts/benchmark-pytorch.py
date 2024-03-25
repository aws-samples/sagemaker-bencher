# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import glob
import json
import argparse
import time
import random

import numpy as np
import webdataset as wds
import torch
from torchvision.transforms import v2 as tvt
from PIL import Image

CHANNEL_NAME = 'train'
DEBUG = True


def identity(x):
    return x


class ModelMock:
    '''Model mock to emulate a computation of a training step'''
    def compute(self, computation_time):
        return time.sleep(computation_time)


class MapDataset(torch.utils.data.Dataset):
    def __init__(self, files, transform=identity):
        self._files = np.array(files)
        self._transform = transform
   
    @staticmethod
    def _get_label(file):
        return file.split(os.path.sep)[-2]
    
    @staticmethod
    def _read(file):
        return Image.open(file).convert('RGB')
    
    def __len__(self):
        return len(self._files)
    
    def __getitem__(self, idx):
        file = self._files[idx]
        sample = self._transform(self._read(file))
        label = int(self._get_label(file)) - 1    # Labels in [0, MAX) range
        return sample, label


def debug(message):
    if DEBUG:
        print('[DEBUG] ' + message)


def parse_and_validate_args():
    
    def none_or_int(value):
        if str(value).upper() == 'NONE':
            return None
        return int(value)
    
    def none_or_str(value):
        if str(value).upper() == 'NONE':
            return None
        return str(value)
    
    def str_bool(value):
        if str(value).upper() == 'TRUE':
            return True
        elif str(value).upper() == 'FALSE':
            return False
        else:
            raise TypeError("Must be True or False.")
    
    parser = argparse.ArgumentParser()

    ### Parameters that define dataloader part 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--prefetch_size', type=none_or_int, default=None)
    parser.add_argument('--input_dim', type=int, default=224)
    parser.add_argument('--pin_memory', type=str_bool, default=True)
    parser.add_argument('--batch_drop_last', type=str_bool, default=False)
    parser.add_argument('--compute_time', type=none_or_int, default=None) # in MS
    
    ### NOT YET IMPLEMENTED
    parser.add_argument('--prefetch_to_gpu', type=str_bool, default=False)
    parser.add_argument('--shuffle', type=str_bool, default=False)
    parser.add_argument('--cache', type=none_or_str, default=None)
    parser.add_argument('--backbone_model', type=none_or_str, default=None)
    
    ### Parameters that define computation part 
    parser.add_argument('--epochs', type=int, default=1)
   
    ### Parameters that define some storage and dataset details for benchmarking 
    parser.add_argument('--input_channel', type=str, default=os.environ.get(f'SM_CHANNEL_{CHANNEL_NAME.upper()}'))
    parser.add_argument('--input_mode', type=str, default=os.environ.get(f'INPUT_MODE_CHANNEL_{CHANNEL_NAME.upper()}'))
    parser.add_argument('--dataset_format', type=none_or_str, default=os.environ.get(f'DATASET_FORMAT_CHANNEL_{CHANNEL_NAME.upper()}'))
    parser.add_argument('--dataset_s3_uri', type=str, default=os.environ.get(f'DATASET_S3_URI_CHANNEL_{CHANNEL_NAME.upper()}'))
    parser.add_argument('--dataset_num_classes', type=none_or_int, default=os.environ.get(f'DATASET_NUM_CLASSES_CHANNEL_{CHANNEL_NAME.upper()}', 2))
    parser.add_argument('--dataset_num_samples', type=none_or_int, default=os.environ.get(f'DATASET_NUM_SAMPLES_CHANNEL_{CHANNEL_NAME.upper()}', 0))

    args, _ = parser.parse_known_args()

    if not bool(args.compute_time) ^ bool(args.backbone_model):
        raise ValueError("Either compute time or backbone model alias must be set..")
    
    return args


def build_dataloader(config):

    transform = tvt.Compose([
        tvt.ToImage(),
        tvt.ToDtype(torch.uint8, scale=True),
        tvt.RandomResizedCrop(size=(config.input_dim, config.input_dim), antialias=True),
        tvt.ToDtype(torch.float32, scale=True),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Build dataset
    if config.dataset_format == 'jpg':
        files = glob.glob(config.input_channel + '/**/*.jpg')
        dataset = MapDataset(files, transform)
        dataloader_batch_size = config.batch_size
    elif config.dataset_format == 'tar/jpg':
        files = glob.glob(config.input_channel + '/*.tar')
        dataset = (
            wds.WebDataset(files, shardshuffle=True)
            .decode('pil')
            .to_tuple('jpg', 'cls')
            .map_tuple(transform, identity)
            .batched(config.batch_size))
        dataloader_batch_size = None
    else:
        raise NotImplementedError("Unknown dataset format '%s'.." % config.dataset_format)
    
    # Debug
    debug("Total files: %d" % len(files))
    debug("First 5 files:")
    for i, f in enumerate(files):
        if i > 5: break
        debug('- %s' % f)

    # Build dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=config.num_workers,
        batch_size=dataloader_batch_size,
        prefetch_factor=config.prefetch_size,
        pin_memory=config.pin_memory
    )

    return dataloader


def build_model(cfg):
    if cfg.compute_time is not None:
        model = _build_model_mock(cfg)
    elif cfg.backbone_model is not None:
        model = _build_model(cfg)
    return model

def _build_model_mock(cfg):
    model = ModelMock()
    return model

def _build_model(cfg):
    raise NotImplementedError("Not yet implemented..")

def train_model(model, dataloader, cfg):
    if cfg.compute_time is not None: 
        stats = _train_model_mock(model, dataloader, cfg)
    elif cfg.backbone_model is not None:
        stats = _train_model(model, dataloader, cfg)
    return stats
    
def _train_model_mock(model, dataloader, cfg):
    t_stats, img_tot_list, ep_times = {}, [], []
    t_train_start = t_epoch_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        img_tot = 0
        for iteration, (images, labels) in enumerate(dataloader, 1):
            # do training step
            batch_size = images.shape[0]
            img_tot += batch_size
            if cfg.compute_time > 0:
                model.compute(cfg.compute_time/1000) # ms --> s

        # log metrics
        img_tot_list.append(img_tot)
        ep_times.append(time.perf_counter() - t_epoch_start)
        t_epoch_start = time.perf_counter()

    # log metrics
    t_train_tot = time.perf_counter() - t_train_start
    t_stats['t_training_exact'] = t_train_tot
    t_stats['img_sec_ave_tot'] = sum(img_tot_list) / t_train_tot
    t_stats['img_tot'] = sum(img_tot_list)
    t_stats.update({f't_epoch_{ep}': t for ep, t in enumerate(ep_times, 1)})
    return t_stats

def _train_model(model, dataloader, cfg):
    raise NotImplementedError("Not yet implemented..")



if __name__ == "__main__":

    from smexperiments.tracker import Tracker

    # Step 1: Parse the parameters sent by the SageMaker client to the script
    args = parse_and_validate_args()

    print("Benchmarking params:\n" + json.dumps(vars(args), indent=2))

    # Step 2: Load SageMaker Experiment tracker to log benchmark metrics
    tracker = Tracker.load()
    
    # Step 3: Build dataloader
    dataloader = build_dataloader(args)

    # Step 4: Build model
    model = build_model(args)

    # Step 5: Do training run
    metrics = train_model(model, dataloader, args)

    print("All logged metrics:\n" + json.dumps(metrics, indent=2))

    # Step 6: Flush logged metrics to SageMaker Experiments
    tracker.log_parameters(metrics)
    
    time.sleep(5)
    tracker.close()
    time.sleep(5)