# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


import concurrent.futures as cf
import os
import shutil
import tempfile

import tqdm
import numpy as np
import tensorflow as tf

import sagemaker_bencher.utils as utils
from .dataset import BenchmarkDataset


class SyntheticBenchmarkDataset(BenchmarkDataset):
    """A collection of synthetic multi-class vector data for training stored in Protobuf-files.

    Each dataset contains multiple records of labeled floating point vectors. Each vector
    is sampled from a Gaussian distribution with a class specific mean. This means it should
    be possible to build a classifier that performs better than random guessing using
    datasets built wtih this class.

    Datasets are generated locally and uploaded to s3 in a specific bucket with specific
    prefix.
    """

    input_formats = {'tfrecord/raw'}

    def __init__(self, name, format, bucket, prefix, region,
                 dimension, num_records, num_files, num_copies, num_classes):
        """Create a SyntheticBenchmarkDataset.

        Args:
            dimension (int): The number of features in the dataset.
            num_records (int): How many records to write in each dataset file.
            num_files (int): How many distinct files of unique labeled records to create.
            num_copies (int): How many times to duplicate each file when being uploaded to s3.
            num_classes (int): How many classes to generate.
            (+ other args to parent class)
        """
        
        self.dimension = dimension
        self.num_records = num_records
        self.num_files = num_files
        self.num_copies = num_copies
        self.num_classes = num_classes
        self.num_samples = num_files * num_copies * num_records

        if format not in self.input_formats:
            raise NotImplementedError("Implemented input formats are '{}', but '{}' requested!..").format(
                self.input_formats, format)

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
        self._make_benchmark_files()
        self._upload_to_s3()
        self._cleanup()

    def _make_benchmark_files(self):
        for file_index in range(self.num_files):
            print("[{}] Generating TFRecord file {} of {}..".format(self, file_index + 1, self.num_files))
            tf_filename = os.path.join(self.root_dir, '{}-{}.tfrecord'.format(self.name, str(file_index)))
            self._build_record_file(tf_filename)

    def _build_record_file(self, filename, verbose=False):
        """Build a TFRecord encoded file of TF protobuf Example objects.

        Each object is a labeled numpy array. Each example has two field - a single int64 'label'
        field and a single bytes list field, containing a serialized flattened numpy array.

        Each generated numpy array is a multidimensional normal with
        the specified dimension. The normal distribution is class specific, each class
        has a different mean for the distribution, so it should be possible to learn
        a multiclass classifier on this data. Class means are determnistic - so multiple
        calls to this function with the same number of classes will produce samples drawn
        from the same distribution for each class.

        Args:
            filename (str): the file to write to.
        """

        with tf.io.TFRecordWriter(filename) as f:
            for i in tqdm.tqdm(range(self.num_records), disable=not verbose):
                label = i % self.num_classes
                loc = 255 * (label + 1) / (self.num_classes + 1)
                arr = np.random.normal(loc=loc, scale=2, size=self.dimension)
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                BenchmarkDataset._write_image_raw(f, arr, label)

    def _upload_to_s3(self):
        print("[{}] Uploading dataset to '{}'..".format(self, self.s3_uri))

        futures = []
        uploaded_file_index = 0
        with tqdm.tqdm(total=self.num_files*self.num_copies) as pbar:
            with cf.ProcessPoolExecutor(2 * os.cpu_count()) as executor:
                for file_index in range(self.num_files):
                    for copy_index in range(self.num_copies):
                        local_file = os.path.join(self.root_dir, '{}-{}.tfrecord'.format(self.name, str(file_index)))
                        remote_key = '{}/{}/file_{}.tfrecord'.format(self.prefix, self.name, str(uploaded_file_index).zfill(6))
                        futures.append(executor.submit(utils.upload_file, self.bucket_name, local_file, remote_key))
                        uploaded_file_index += 1

                for f in cf.as_completed(futures):
                    try:
                        _ = f.result()
                        pbar.update(1)
                    except Exception as e:
                        print(e)

    def _cleanup(self):
        shutil.rmtree(self.root_dir)
