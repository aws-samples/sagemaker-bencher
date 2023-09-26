# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import json
import argparse
import time

import tensorflow as tf

from smexperiments.tracker import Tracker
from sagemaker_tensorflow import PipeModeDataset
from classification_models.tfkeras import Classifiers

CHANNEL_NAME = 'train'
DEBUG = True


def debug(message):
    if DEBUG:
        print('[DEBUG] ' + message)


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self.epoch_time_start)


def _parse_args():
    
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

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--prefetch_first', type=str_bool, default=False)
    parser.add_argument('--prefetch_size', type=none_or_int, default=None)
    parser.add_argument('--prefetch_to_gpu', type=str_bool, default=False)
    parser.add_argument('--num_parallel_calls', type=none_or_int, default=None)
    parser.add_argument('--num_parallel_reads', type=none_or_int, default=None)
    parser.add_argument('--private_pool', type=none_or_int, default=None)
    parser.add_argument('--backbone_model', type=str)
    parser.add_argument('--input_dim', type=int, default=224)
    parser.add_argument('--cache', type=none_or_str, default=None)
    parser.add_argument('--parallel_batch', type=str_bool, default=False)
   
    # other benchmark params
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--num_gpus', type=str, default=os.environ.get('SM_NUM_GPUS'))
    parser.add_argument('--framework', type=str, default=os.environ.get('IO_FRAMEWORK'))
    parser.add_argument('--input_channel', type=str, default=os.environ.get(f'SM_CHANNEL_{CHANNEL_NAME.upper()}'))
    parser.add_argument('--input_mode', type=str, default=os.environ.get(f'IO_INPUT_MODE_CHANNEL_{CHANNEL_NAME.upper()}'))
    parser.add_argument('--input_format', type=none_or_str, default=os.environ.get(f'IO_INPUT_FORMAT_CHANNEL_{CHANNEL_NAME.upper()}'))
    parser.add_argument('--s3_data_source', type=str, default=os.environ.get(f'IO_DATA_SOURCE_CHANNEL_{CHANNEL_NAME.upper()}'))
    parser.add_argument('--num_classes', type=none_or_int, default=os.environ.get(f'IO_NUM_CLASSES_CHANNEL_{CHANNEL_NAME.upper()}', 2))
    parser.add_argument('--num_samples', type=none_or_int, default=os.environ.get(f'IO_NUM_SAMPLES_CHANNEL_{CHANNEL_NAME.upper()}', 0))
    
    return parser.parse_known_args()


def _build_tf_dataloader(config):

    t_stats = {}
    t_import_framework = time.perf_counter()
    
    t_stats['t_import_framework'] = time.perf_counter() - t_import_framework

    features_raw = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64)}

    features_jpg = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)}

    def parse_record_raw(record):
        parsed = tf.io.parse_single_example(record, features_raw)
        shape = [parsed['height'], parsed['width'], parsed['depth']]
        image = tf.io.decode_raw(parsed['image'], tf.uint8)
        image = tf.reshape(image, shape)
        image = tf.image.resize(image, (config.input_dim, config.input_dim))
        label = parsed['label'] - 1    # Need labels in [0, MAX) range
        return image, label

    def parse_record_jpg(record):
        parsed = tf.io.parse_single_example(record, features_jpg)
        image = tf.io.decode_jpeg(parsed['image'], channels=3)
        image = tf.image.resize(image, (config.input_dim, config.input_dim))
        label = parsed['label'] - 1    # Need labels in [0, MAX) range
        return image, label

    def parse_file_jpg(file):
        image, label = read_file_jpg(file)
        image, label = preproc_file_jpg(image, label)
        return image, label

# ==== FOR CACHING ====
    def read_file_jpg(file):
        image = tf.io.read_file(file)
        label = tf.strings.split(file, sep=os.path.sep)[-2]
        label = tf.strings.to_number(label, tf.int32) - 1    # Need labels in [0, MAX) range
        return image, label

    def preproc_file_jpg(image, label):
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (config.input_dim, config.input_dim))
        return image, label
# ==== END FOR CACHING ====

    if config.input_format == 'jpg':
        search_ex = '/**/*.jpg' # folder structure: './<class_id>/image.jpg'
        parse_fn = parse_file_jpg
    elif config.input_format == 'tfrecord/jpg':
        search_ex = '/*.tfrecord' # folder structure: './data.tfrecord'
        parse_fn = parse_record_jpg
    elif config.input_format == 'tfrecord/raw':
        search_ex = '/*.tfrecord' # folder structure: './data.tfrecord'
        parse_fn = parse_record_raw

    if config.input_mode == 'pipe':
        if config.input_format.startswith('tfrecord'):
            ds = PipeModeDataset(channel=CHANNEL_NAME, record_format='TFRecord', benchmark=True)
        else:
            raise NotImplementedError("Pipe-mode only support TFRecord-input!..")
    elif config.input_mode in {'file', 'ffm', 'fsx'}:
        files = tf.io.gfile.glob(config.input_channel + search_ex)
        if config.input_format.startswith('tfrecord'):
            ds = tf.data.TFRecordDataset(files, num_parallel_reads=config.num_parallel_reads)
        elif config.input_format == 'jpg':
            ds = tf.data.Dataset.from_tensor_slices(files)
        else:
            raise NotImplementedError("File, FSx, and FFM-modes only support TFRecords- or JPG-inputs!..")
    elif config.input_mode == 's3tf':
        if config.input_format.startswith('tfrecord'):
            files = tf.io.gfile.glob(config.s3_data_source + search_ex)
            ds = tf.data.TFRecordDataset(files, num_parallel_reads=config.num_parallel_reads)
        else:
            raise NotImplementedError("S3TF-mode only support TFRecord-input!..")
    else:
        raise NotImplementedError("Not implemented for '%s'.." % args.input_mode)
        
    
        
    if config.prefetch_first:
        debug("Prefetch first elements is ON!..")
        ds = ds.prefetch(-1)

    if config.cache and config.input_format == 'jpg':
        debug('Caching training data to %s!..' % config.cache)
        ds = ds.map(read_file_jpg, num_parallel_calls=config.num_parallel_calls)
        ds = ds.cache('' if config.cache.upper() == 'MEM' else config.cache)
        ds = ds.map(preproc_file_jpg, num_parallel_calls=config.num_parallel_calls)
    else:
        debug('Caching inputs is OFF!..')
        ds = ds.map(parse_fn, num_parallel_calls=config.num_parallel_calls)
        
    if config.parallel_batch:
        debug('Parallel batching is ON!..')
        ds = ds.batch(config.batch_size, num_parallel_calls=-1, deterministic=False)
    else:
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
    
    t_stats['t_build_pipe'] = time.perf_counter() - t_import_framework

    return ds, t_stats


def _build_tf_model(backbone_model, num_classes, input_shape=(224, 224, 3), compiled=True):

    backbone_fn, _ = Classifiers.get(backbone_model)
        
    base_model = backbone_fn(input_shape=input_shape, weights='imagenet', include_top=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=[base_model.input], outputs=[output])
    
    if compiled:
        model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')
        
    return model
    
    


if __name__ == "__main__":

    args, unknown = _parse_args()

    debug("Env args: \n" + json.dumps(vars(args)))
    
    print("Starting benchmark for input mode '%s'.." % args.input_mode)
    tracker = Tracker.load()

    if args.framework == 'tf':
        dataloader, t_stats = _build_tf_dataloader(config=args)
    else:
        raise NotImplementedError("Not implemented for '%s'.." % args.framework)
    
    t_build_model = time.perf_counter()
    model = _build_tf_model(args.backbone_model, args.num_classes, input_shape=(args.input_dim, args.input_dim, 3), compiled=True)
    t_stats['t_build_model'] = time.perf_counter() - t_build_model
    
    t_train_start = time.perf_counter()
    time_callback = TimeHistory()
    
    model.fit(dataloader, epochs=args.epochs, callbacks=[time_callback])

    img_tot = args.num_samples * args.epochs
    t_train_tot = time.perf_counter() - t_train_start
    t_stats['t_training_exact'] = t_train_tot
    t_stats['img_sec_ave_tot'] = img_tot / t_train_tot
    t_stats['img_tot'] = img_tot
    t_stats.update({f't_epoch_{ep}': t for ep, t in enumerate(time_callback.times, 1)})
    tracker.log_parameters(t_stats)
    
    print(json.dumps(t_stats))

    time.sleep(5)
    tracker.close()
    time.sleep(5)
