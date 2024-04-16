# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import glob
import json
import argparse
import time
import random

import numpy as np
import torch
from torchvision.transforms import v2 as tvt
from PIL import Image

import torchdata
import webdataset as wds
import s3torchconnector as s3pt

CHANNEL_NAME = 'train'
DEBUG = True


def debug(message):
    if DEBUG:
        print('[DEBUG] ' + message)


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
    parser.add_argument('--torchdata', type=str_bool, default=False)
    
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
    parser.add_argument('--dataset_s3_region', type=str, default=os.environ.get('SAGEMAKER_REGION'))
    parser.add_argument('--dataset_num_classes', type=none_or_int, default=os.environ.get(f'DATASET_NUM_CLASSES_CHANNEL_{CHANNEL_NAME.upper()}', 2))
    parser.add_argument('--dataset_num_samples', type=none_or_int, default=os.environ.get(f'DATASET_NUM_SAMPLES_CHANNEL_{CHANNEL_NAME.upper()}', 0))

    args, _ = parser.parse_known_args()

    if not (args.compute_time is not None) ^ (args.backbone_model is not None):
        raise ValueError("Either compute time or backbone model alias must be set..")
    
    return args

def _make_pt_dataset(config, transform):
    files = glob.glob(config.input_channel + '/**/*.jpg')
    dataset = MapDataset(files, transform)
    return dataset

def _make_wds_dataset(config, transform):
    files = glob.glob(config.input_channel + '/*.tar')
    dataset = wds.WebDataset(files, shardshuffle=True)
    dataset = dataset.decode('pil')
    dataset = dataset.to_tuple('jpg', 'cls')
    dataset = dataset.map_tuple(transform, identity)
    return dataset

def _make_s3pt_map_dataset(config, transform):

    def _read_and_transform(sample):
        key, data = sample.key, sample
        img = transform(Image.open(data).convert('RGB'))
        label = int(key.split('/')[-2])
        return img, label

    dataset = s3pt.S3MapDataset.from_prefix(
        config.dataset_s3_uri,
        region=config.dataset_s3_region,
        transform=_read_and_transform)
    return dataset

def _make_s3pt_iter_dataset(config, transform):

    def _tar_to_tuple(s3object):
        return s3object.key, torchdata.datapipes.utils.StreamWrapper(s3object)
    
    def _read_and_transform(item, cls_transform=int, jpg_transform=transform):
        key, value = item
        if key.endswith('.cls'):
            return key, cls_transform(value.read())
        if key.endswith('.jpg'):
            return key, jpg_transform(Image.open(value).convert('RGB'))
    
    def _wds_to_tuple(item):
        return item['.jpg'], item['.cls']

    dataset = s3pt.S3IterableDataset.from_prefix(
        config.dataset_s3_uri,
        region=config.dataset_s3_region)
    dataset = torchdata.datapipes.iter.IterableWrapper(dataset)
    if config.num_workers > 0:
        dataset = dataset.sharding_filter()
    dataset = dataset.map(_tar_to_tuple)
    dataset = dataset.load_from_tar() 
    dataset = dataset.map(_read_and_transform)
    dataset = dataset.webdataset()
    dataset = dataset.map(_wds_to_tuple)
    return dataset

def _make_fsspec_map_dataset(config, transform):
    
    def _read_and_transform(sample):
        key, data = sample
        img = transform(Image.open(data).convert('RGB'))
        label = int(key.split('/')[-2])
        return img, label
    
    s3_prefixes = list(torchdata.datapipes.iter.FSSpecFileLister(config.dataset_s3_uri))  # Trick to get list of "subdirs"
    dataset = torchdata.datapipes.iter.FSSpecFileLister(s3_prefixes)
    if config.num_workers > 0:
        dataset = dataset.sharding_filter()
    dataset = dataset.open_files_by_fsspec(mode='rb')
    dataset = dataset.map(_read_and_transform)
    return dataset


def _make_fsspec_iter_dataset(config, transform):

    def _read_and_transform(item, cls_transform=int, jpg_transform=transform):
        key, value = item
        if key.endswith('.cls'):
            return key, cls_transform(value.read())
        if key.endswith('.jpg'):
            return key, jpg_transform(Image.open(value).convert('RGB'))
        
    def _wds_to_tuple(item):
        return item['.jpg'], item['.cls']
    
    dataset = torchdata.datapipes.iter.FSSpecFileLister(config.dataset_s3_uri)
    if config.num_workers > 0:
        dataset = dataset.sharding_filter()
    dataset = dataset.open_files_by_fsspec(mode='rb')
    dataset = dataset.load_from_tar() 
    dataset = dataset.map(_read_and_transform)
    dataset = dataset.webdataset()
    dataset = dataset.map(_wds_to_tuple)
    return dataset


def build_dataloader(config):

    transform = tvt.Compose([
        tvt.ToImage(),
        tvt.ToDtype(torch.uint8, scale=True),
        tvt.RandomResizedCrop(size=(config.input_dim, config.input_dim), antialias=False), #antialias=True
        tvt.ToDtype(torch.float32, scale=True),
        #tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Build dataset
    if config.input_mode in {'file', 'ffm'}:
        if config.dataset_format == 'jpg':
            dataset = _make_pt_dataset(config, transform)
        elif config.dataset_format == 'tar/jpg':
            dataset = _make_wds_dataset(config, transform)
        else:
            raise NotImplementedError("Unknown dataset format '%s'.." % config.dataset_format)
    elif config.input_mode == 's3pt':
        if config.dataset_format == 'jpg':
            dataset = _make_s3pt_map_dataset(config, transform)
        elif config.dataset_format == 'tar/jpg':
            dataset = _make_s3pt_iter_dataset(config, transform)
    elif config.input_mode == 'fsspec':
        if config.dataset_format == 'jpg':
            dataset = _make_fsspec_map_dataset(config, transform)
        elif config.dataset_format == 'tar/jpg':
            dataset = _make_fsspec_iter_dataset(config, transform)
    elif config.input_mode == 's3wds':
        if config.dataset_format == 'jpg':
            raise NotImplementedError("Input mode '%s' supports on 'tar/wds'.." % config.input_mode)
        elif config.dataset_format == 'tar/jpg':
            # ...
            pass
    else:
        raise NotImplementedError("Input mode '%s' is not implemented.." % config.input_mode)

    
    # Debug
    #debug("Total files: %d" % len(files))
    #debug("First 5 files:")
    #for i, f in enumerate(files):
    #    if i > 5: break
    #    debug('- %s' % f)

    # Build dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
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
            print(iteration, '-->', images.shape, labels.shape, labels)

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