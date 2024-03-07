# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


import io
import os
import random
import shutil
import tempfile
import shortuuid

import tqdm
import numpy as np
from PIL import Image as pil_image

import tensorflow as tf
import webdataset as wds

from sagemaker_bencher import utils
from sagemaker_bencher.dataset import BenchmarkDataset


class SyntheticBenchmarkDataset(BenchmarkDataset):
    """A collection of synthetic multi-class vector data for training stored in Protobuf-files.

    Each dataset contains multiple records of labeled floating point vectors. Each vector
    is sampled from a Gaussian distribution with a class specific mean. This means it should
    be possible to build a classifier that performs better than random guessing using
    datasets built wtih this class.

    Datasets are generated locally and uploaded to s3 in a specific bucket with specific
    prefix.
    """

    input_formats = {'tar/jpg', 'tfrecord/raw', 'tfrecord/jpg', 'jpg'}

    def __init__(self, name, format, type, bucket, prefix, region, dimension,
                 num_files, num_copies, num_classes, num_records=None):
        """Create a SyntheticBenchmarkDataset.

        Args:
            name (str): The name of the dataset.
            format (str): The format of the dataset.
            type (str): The type of the dataset.
            bucket (str): The name of the S3 bucket where the dataset will be stored.
            prefix (str): The prefix within the S3 bucket.
            region (str): The AWS region where the S3 bucket is located.
            dimension (int): The number of features in the dataset.
            num_records (int, optional): How many records to write in each dataset file.
            num_files (int): How many distinct files of unique labeled records to create.
            num_copies (int): How many times to duplicate each file when being uploaded to S3.
            num_classes (int): How many classes to generate.
        """

        if format not in self.input_formats:
            raise NotImplementedError("Implemented input formats are '{}', but '{}' requested!..").format(
                self.input_formats, format)
        
        self.dimension = dimension
        self.num_records = num_records
        self.num_files = num_files
        self.num_copies = num_copies
        self.num_classes = num_classes
        self.num_samples = num_files * num_copies * (num_records or 1)
        self.verbose = True

        super().__init__(name, format=format, type=type, bucket=bucket, prefix=prefix, region=region)


    def build(self, overwrite=False):
        """Build the dataset and upload to s3.

        Args:
            overwrite (bool): If true will overwrite the dataset if it exists already.
        """
        
        self.bucket_name = utils.get_bucket(bucket=self.bucket_name, region=self.region)
        
        if self._exists() and not overwrite:
            print(f"Dataset '{self.name}' found on '{self.s3_uri}'. Skipping build..")
            return
        else:
            print(f"Building dataset '{self.name}'..")
            
        self.root_dir = tempfile.mkdtemp() # tempfile.mkdtemp(prefix=self.name + '-', dir='test-build')
        benchmark_files, remote_subdirs = self._make_benchmark_files()
        utils.upload_dataset(self, benchmark_files, remote_subdirs)
        self._cleanup()


    def _make_benchmark_files(self):
        print(f"[{self}] Staging dataset in '{self.root_dir}'..")
        if self.format.startswith('tfrecord'):
            img_list, label_list = self._make_record_files()
        elif self.format.startswith('tar'):
            img_list, label_list = self._make_tar_files()
        else:
            img_list, label_list = self._make_jpg_files()
        return img_list, label_list
    

    def _make_tar_files(self):
        tar_list = []
        for file_index in range(self.num_files):
            print("[{}] Generating TAR file {} of {}..".format(self, file_index + 1, self.num_files))
            tar_filename = os.path.join(self.root_dir, '{}-{:06d}.tar'.format(self.name, file_index))
            tar_list.append(tar_filename)

            with wds.TarWriter(tar_filename) as f:
                for _ in tqdm.tqdm(range(self.num_records), disable=not self.verbose):
                    arr, label = self._gen_sample()
                    img_bytes = self._encode_image(arr, format='JPEG')
                    key = os.path.join(str(label), 'sample-%s' % shortuuid.uuid())
                    sample = {"__key__": key, "jpg": img_bytes, "cls": label}
                    f.write(sample)

        return tar_list, None


    def _make_record_files(self):
        tfr_list = []
        for file_index in range(self.num_files):
            print("[{}] Generating TFRecord file {} of {}..".format(self, file_index + 1, self.num_files))
            tfr_filename = os.path.join(self.root_dir, '{}-{}.tfrecord'.format(self.name, str(file_index)))
            tfr_list.append(tfr_filename)

            with tf.io.TFRecordWriter(tfr_filename) as f:
                for _ in tqdm.tqdm(range(self.num_records), disable=not self.verbose):
                    arr, label = self._gen_sample()
                    if self.format.endswith('raw'):
                        BenchmarkDataset._write_image_raw(f, arr, label)
                    else:
                        img_bytes = self._encode_image(arr, format='JPEG')
                        BenchmarkDataset._write_image_bytes(f, img_bytes, label)

        return tfr_list, None
    

    def _make_jpg_files(self):
        img_list, label_list = [], []
        for c in range(self.num_classes):
            os.makedirs(os.path.join(self.root_dir, str(c)), exist_ok=True)
        for _ in tqdm.tqdm(range(self.num_files), disable=not self.verbose):
            arr, label = self._gen_sample()
            img_bytes = self._encode_image(arr, format='JPEG')
            img_filename = os.path.join(self.root_dir, str(label), 'sample-%s.jpg' % shortuuid.uuid())
            with open(img_filename, 'wb') as f: 
                f.write(img_bytes)
            img_list.append(img_filename)
            label_list.append(label)
        return img_list, label_list
    
    
    @staticmethod
    def _encode_image(arr, format):
        """
        Encode an image from a NumPy array and return it as a byte array.

        Args:
            arr (numpy.ndarray): The NumPy array representing the image.
            format (str): The image format (e.g., 'JPEG', 'PNG').

        Returns:
            bytes: A byte array containing the encoded image.
        """
        img_byte_arr = io.BytesIO()
        img = pil_image.fromarray(arr)
        img.save(img_byte_arr, format=format, subsampling=0, quality=100)
        return img_byte_arr.getvalue()
    
    
    def _gen_sample(self):
        """
        Generate a random sample.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The generated data sample.
                - int: The label associated with the sample.
        """
        label = random.randrange(self.num_classes)
        loc = 255 * (label + 1) / (self.num_classes + 1)
        arr = np.random.normal(loc=loc, scale=2, size=self.dimension)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr, label
    

    def _cleanup(self):
        shutil.rmtree(self.root_dir)


if __name__ == '__main__':

    region = 'us-west-2'

    dataset = SyntheticBenchmarkDataset(
        name='test2',
        format='tar/jpg',
        type='synthetic',
        bucket=f'sagemaker-benchmark-{region}-XXXXXXXXXXX',
        prefix='datasets/synth-test',
        region=region,
        dimension=(288, 288, 3),
        num_records=1000,
        num_files=10,
        num_copies=2,
        num_classes=4
    )

    dataset.build(overwrite=True)
