# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import json
import argparse
import time
import random

import tensorflow as tf

CHANNEL_NAME = 'train'
DEBUG = True


class TFModelMock:
    '''This is a mock of TF model to emulate a computation of a training step'''
    
    def model(epoch, computation_time):
        time.sleep(computation_time)

    def compute(self, computation_time):
        tf.function(self.model)(computation_time)


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self.epoch_time_start)


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
    parser.add_argument('--prefetch_first', type=str_bool, default=False)
    parser.add_argument('--prefetch_size', type=none_or_int, default=None)
    parser.add_argument('--prefetch_to_gpu', type=str_bool, default=False)
    parser.add_argument('--num_parallel_calls', type=none_or_int, default=None)
    parser.add_argument('--num_parallel_reads', type=none_or_int, default=None)
    parser.add_argument('--private_pool', type=none_or_int, default=None)
    parser.add_argument('--input_dim', type=int, default=224)
    parser.add_argument('--shuffle', type=str_bool, default=False)
    parser.add_argument('--cache', type=none_or_str, default=None)
    
    ### Parameters that define computation part 
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--compute_time', type=none_or_int, default=None) # in MS
    parser.add_argument('--backbone_model', type=none_or_str, default=None)
   
    ### Parameters that define some storage and dataset details for benchmarking 
    parser.add_argument('--input_channel', type=str, default=os.environ.get(f'SM_CHANNEL_{CHANNEL_NAME.upper()}'))
    parser.add_argument('--input_mode', type=str, default=os.environ.get(f'INPUT_MODE_CHANNEL_{CHANNEL_NAME.upper()}'))
    parser.add_argument('--dataset_format', type=none_or_str, default=os.environ.get(f'DATASET_FORMAT_CHANNEL_{CHANNEL_NAME.upper()}'))
    parser.add_argument('--dataset_s3_uri', type=str, default=os.environ.get(f'DATASET_S3_URI_CHANNEL_{CHANNEL_NAME.upper()}'))
    parser.add_argument('--dataset_num_classes', type=none_or_int, default=os.environ.get(f'DATASET_NUM_CLASSES_CHANNEL_{CHANNEL_NAME.upper()}', 2))
    parser.add_argument('--dataset_num_samples', type=none_or_int, default=os.environ.get(f'DATASET_NUM_SAMPLES_CHANNEL_{CHANNEL_NAME.upper()}', 0))

    args, _ = parser.parse_known_args()

    if args.compute_time is not None and args.backbone_model is not None:
        raise ValueError("Compute time and backbone model can't be set together..")
    
    return args


def build_dataloader(config):

    # Define signature of TFRecord samples in raw format
    features_raw = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64)}

    # Define signature of TFRecord samples in JPG format
    features_jpg = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)}

    # Define parsing and preproc function for TFRecord samples in raw format
    def parse_record_raw(record):
        parsed = tf.io.parse_single_example(record, features_raw)
        shape = [parsed['height'], parsed['width'], parsed['depth']]
        image = tf.io.decode_raw(parsed['image'], tf.uint8)
        image = tf.reshape(image, shape)
        image = tf.image.resize(image, (config.input_dim, config.input_dim))
        label = parsed['label'] - 1    # Need labels in [0, MAX) range
        return image, label

    # Define parsing and preproc function for TFRecord samples in JPG format
    def parse_record_jpg(record):
        parsed = tf.io.parse_single_example(record, features_jpg)
        image = tf.io.decode_jpeg(parsed['image'], channels=3)
        image = tf.image.resize(image, (config.input_dim, config.input_dim))
        label = parsed['label'] - 1    # Need labels in [0, MAX) range
        return image, label

    # Define parsing and preproc function for JPG samples
    def parse_file_jpg(file):
        image, label = read_file_jpg(file)
        image, label = preproc_file_jpg(image, label)
        return image, label

    def read_file_jpg(file):
        image = tf.io.read_file(file)
        label = tf.strings.split(file, sep=os.path.sep)[-2]
        label = tf.strings.to_number(label, tf.int32) - 1    # Need labels in [0, MAX) range
        return image, label

    def preproc_file_jpg(image, label):
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (config.input_dim, config.input_dim))
        return image, label

    # Select appropriate parsing function and search expression for either JPG- or TFRecord-datasets
    if config.dataset_format == 'jpg':
        search_ex = '/**/*.jpg' # folder structure: './<class_id>/image.jpg'
        parse_fn = parse_file_jpg
    elif config.dataset_format == 'tfrecord/jpg':
        search_ex = '/*.tfrecord' # folder structure: './data.tfrecord'
        parse_fn = parse_record_jpg
    elif config.dataset_format == 'tfrecord/raw':
        search_ex = '/*.tfrecord' # folder structure: './data.tfrecord'
        parse_fn = parse_record_raw
    else:
        raise NotImplementedError("Unknown dataset format '%s'.." % config.dataset_format)

    # Start building a dataloader pipeline
    if config.input_mode == 'pipe':
        from sagemaker_tensorflow import PipeModeDataset
        if config.dataset_format.startswith('tfrecord'):
            ds = PipeModeDataset(channel=CHANNEL_NAME, record_format='TFRecord', benchmark=True)
        else:
            raise NotImplementedError("Pipe-mode only support TFRecord-input!..")
    elif config.input_mode in {'file', 'ffm', 'fsx'}:
        files = tf.io.gfile.glob(config.input_channel + search_ex)
        if config.dataset_format.startswith('tfrecord'):
            ds = tf.data.TFRecordDataset(files, num_parallel_reads=config.num_parallel_reads)
        elif config.dataset_format == 'jpg':
            ds = tf.data.Dataset.from_tensor_slices(files)
        else:
            raise NotImplementedError("File- and FFM-modes only support TFRecords- or JPG-inputs!..")
    elif config.input_mode == 's3tf':
        if config.dataset_format.startswith('tfrecord'):
            files = tf.io.gfile.glob(config.dataset_s3_uri + search_ex)
            ds = tf.data.TFRecordDataset(files, num_parallel_reads=config.num_parallel_reads)
        else:
            raise NotImplementedError("S3TF-mode only support TFRecord-input!..")
    else:
        raise NotImplementedError("Not implemented for '%s'.." % args.input_mode)
        
    if config.shuffle:
        debug("Shuffling is ON!..")
        random.shuffle(files)
        
    if config.prefetch_first:
        debug("Prefetch first elements is ON!..")
        ds = ds.prefetch(-1)

    if config.cache and config.dataset_format == 'jpg':
        debug('Caching training data to %s!..' % config.cache)
        ds = ds.map(read_file_jpg, num_parallel_calls=config.num_parallel_calls)
        ds = ds.cache('' if config.cache.upper() == 'MEM' else config.cache)
        ds = ds.map(preproc_file_jpg, num_parallel_calls=config.num_parallel_calls)
    else:
        debug('Caching inputs is OFF!..')
        ds = ds.map(parse_fn, num_parallel_calls=config.num_parallel_calls)
        
    ds = ds.batch(config.batch_size)
    
    if config.prefetch_size:
        debug("Batch prefetch is set to %d.." % config.prefetch_size)
        if config.prefetch_to_gpu:
            debug("Batch copy to GPU-0 enabled..")
            ds = ds.apply(tf.data.experimental.copy_to_device("/gpu:0"))
            #ds = ds.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
        ds = ds.prefetch(config.prefetch_size)
    
    if config.private_pool:
        debug('Using private thread pool of %d size..' % config.private_pool)
        options = tf.data.Options()
        options.experimental_threading.private_threadpool_size = config.private_pool
        ds = ds.with_options(options)

    return ds


def build_model(cfg):
    if cfg.compute_time is not None:
        model = _build_model_mock(cfg)
    elif cfg.backbone_model is not None:
        model = _build_model_backbone(cfg)
    return model


def train_model(model, cfg):
    if cfg.compute_time is not None: 
        stats = _train_model_mock(model, cfg)
    elif cfg.backbone_model is not None:
        stats = _train_model_backbone(model, cfg)
    return stats


def _build_model_mock(cfg):
    model = TFModelMock()
    return model


def _build_model_backbone(cfg):
    from classification_models.tfkeras import Classifiers

    backbone_fn, _ = Classifiers.get(cfg.backbone_model)
    input_shape = (cfg.input_dim, cfg.input_dim, 3)

    base_model = backbone_fn(input_shape=input_shape, weights='imagenet', include_top=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(cfg.dataset_num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=[base_model.input], outputs=[output])
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')
    return model
    

def _train_model_mock(model, cfg):
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


def _train_model_backbone(model, cfg):
    t_stats = {}
    time_callback = TimeHistory()
    t_train_start = time.perf_counter()

    model.fit(dataloader, epochs=cfg.epochs, callbacks=[time_callback])

    img_tot = cfg.dataset_num_samples * cfg.epochs
    t_train_tot = time.perf_counter() - t_train_start
    t_stats['t_training_exact'] = t_train_tot
    t_stats['img_sec_ave_tot'] = img_tot / t_train_tot
    t_stats['img_tot'] = img_tot
    t_stats.update({f't_epoch_{ep}': t for ep, t in enumerate(time_callback.times, 1)})
    return t_stats


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
    metrics = train_model(model, args)
    
    print("All logged metrics:\n" + json.dumps(metrics, indent=2))
    
    # Step 6: Flush logged metrics to SageMaker Experiments
    tracker.log_parameters(metrics)
    
    time.sleep(5)
    tracker.close()
    time.sleep(5)
