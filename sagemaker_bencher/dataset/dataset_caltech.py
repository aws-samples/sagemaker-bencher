# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


import concurrent.futures as cf
import os
import tarfile
import shutil
import tempfile

import tqdm
import tensorflow as tf
import numpy as np
from PIL import Image 

import sagemaker_bencher.utils as utils
from .dataset import BenchmarkDataset


class CaltechBenchmarkDataset(BenchmarkDataset):

    input_formats = {'tfrecord/raw', 'tfrecord/jpg', 'jpg'}

    def __init__(self, name, format, bucket, prefix, region, num_copies, num_records=None):
        """Create a SyntheticBenchmarkDataset.

        Args:
            name (str): The name of the dataset
            bucket (str): An S3 bucket to store the dataset in
            prefix (str): An S3 prefix directory to store dataset objects in, within the bucket
            format (str): either 'tfrecord/raw', 'tfrecord/jpg', or 'jpg'
            num_records (int): How many records to write in each dataset file (applies only to format='tfrecord').
            num_copies (int): How many times to duplicate each file when being uploaded to s3.
        """

        if format not in self.input_formats:
            raise NotImplementedError("Implemented input formats are '{}', but '{}' requested!..").format(
                self.input_formats, format)
        
        if format.startswith('tfrecord'):
            if num_records is None:
                raise ValueError("For tfrecord-based formats, the 'num_records' must be provided!..")
            if len(format.split('/')) != 2:
                raise ValueError("The format-argument must be of form 'tfrecord/<compression>'!..")

        self.num_copies = num_copies
        self.num_records = num_records
        self.verbose = False
        self.num_classes = 257
        self.num_samples = 30607 * num_copies

        super().__init__(name, format, bucket=bucket, prefix=prefix, region=region)


    def build(self, overwrite=False):
        """Build the dataset and upload to s3.

        Args:
            overwrite (bool): If true will overwrite the dataset if it exists already.
        """
        if self._exists() and not overwrite:
            print(f"Dataset '{self.name}' found on '{self.s3_uri}'. Skipping build..")
            return
        else:
            print(f"Building dataset '{self.name}'..")
        self.root_dir = tempfile.mkdtemp()
        self._download_from_source(untar=True)
        img_list, label_list = self._parse_dataset(file_extensions=['.jpg'])

        benchmark_files, remote_subdirs = self._make_benchmark_files(img_list, label_list)
        self._upload_to_s3(benchmark_files, remote_subdirs)
        self._cleanup()
        

    def _download_from_source(self, untar=False):
        caltech_gd = '1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK'
        caltech_md5 = '67b4f42ca05d46448c6bb8ecd2220f6d'
        caltech_file = 'caltech256.tar'

        print(f"Downloading '{caltech_gd}' into '{os.path.join(self.root_dir, caltech_file)}'..")
        utils.download_file_from_google_drive(caltech_gd, self.root_dir, caltech_file)

        if untar:
            tar = tarfile.open(os.path.join(self.root_dir, caltech_file))
            tar.extractall(self.root_dir)
            tar.close()


    def _parse_dataset(self, file_extensions=None, sort=False):

        def _parse_label(path):
            x = os.path.normpath(path)
            x = os.path.basename(x)
            x = int(x[:3])
            return x

        img_list = []
        label_list = []

        img_paths = utils.get_files_from_dir(self.root_dir, file_extensions, sort)
        for img in img_paths:
            lab = _parse_label(img)
            img_list.append(img)
            label_list.append(lab)

        return img_list, label_list


    def _make_benchmark_files(self, img_list, label_list):
        assert len(img_list) == len(label_list), "Length of images and labels must match!.."

        if self.format == 'jpg':
            return img_list, label_list
        elif self.format.startswith('tfrecord'):
            tfr_list = []
            for inx in range(0, len(img_list), self.num_records):
                # if len(tfr_list) > 0: break   # DEBUGGING / TESTING
                tfr_path = os.path.join(self.root_dir, 'data-{:06d}.tfrecord'.format(len(tfr_list)))
                print("[{}] Generating TFRecord file '{}'..".format(self, os.path.basename(tfr_path)))
                inx_slice = slice(inx, inx + self.num_records)
                self._build_record_file(tfr_path, img_list[inx_slice], label_list[inx_slice], self.format)
                tfr_list.append(tfr_path)
            return tfr_list, None


    def _build_record_file(self, filename, image_list, label_list, format):

        def _read_bytes(file):
            with open(file, 'rb') as img_file:
                img_bytes = img_file.read()
                return img_bytes
        
        def _read_jpg(file):
            with Image.open(file) as img_file:
                img_arr = np.asarray(img_file, np.uint8)
                try:
                    img_arr.shape[2]
                except IndexError:
                    img_arr = np.stack((img_arr,) * 3, axis=-1) # Convert grayscale to RGB
                return img_arr

        _, comp = format.split('/')
        if comp == 'raw':
            _read_fn = _read_jpg
            _write_fn = BenchmarkDataset._write_image_raw
        elif comp == 'jpg':
            _read_fn = _read_bytes
            _write_fn = BenchmarkDataset._write_image_bytes
        else:
            raise NotImplementedError("Can't build dataset with '%s' format!..")

        with tf.io.TFRecordWriter(filename) as f:
            for image, label in tqdm.tqdm(zip(image_list, label_list), total=len(image_list), disable=not self.verbose):
                _write_fn(f, _read_fn(image), label)


    def _upload_to_s3(self, local_files, remote_subdir=None):
        print("[{}] Uploading dataset to '{}'..".format(self, self.s3_uri))

        futures = []
        with tqdm.tqdm(total=len(local_files)*self.num_copies) as pbar:
            with cf.ProcessPoolExecutor(2 * os.cpu_count()) as executor:
                for i, file_path in enumerate(local_files):
                    file_base = os.path.basename(file_path)
                    file_name, file_ext = os.path.splitext(file_base)
                    for copy_index in range(self.num_copies):
                        remote_base = '{}-{:03d}{}'.format(file_name, copy_index, file_ext)
                        remote_subpath = '{:03d}/{}'.format(remote_subdir[i], remote_base) if remote_subdir else remote_base
                        remote_key = '{}/{}/{}'.format(self.prefix, self.name, remote_subpath)
                        futures.append(executor.submit(utils.upload_file, self.bucket_name, file_path, remote_key))

                for f in cf.as_completed(futures):
                    try:
                        _ = f.result()
                        pbar.update(1)
                    except Exception as e:
                        print(e)


    def _cleanup(self):
        shutil.rmtree(self.root_dir)