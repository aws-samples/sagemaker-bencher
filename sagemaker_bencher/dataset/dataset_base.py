# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


from abc import ABC, abstractmethod
import tensorflow as tf

from sagemaker_bencher import utils


class Dataset(ABC):
    def __init__(self, name, format=None, type=None, bucket=None, prefix=None, region=None):
        """Base constructor for Dataset.

        Args:
            name (str): The name of the dataset
            format (str): (optional) The format of the dataset (e.g. 'jpg', 'tfrecord')
            type (str): (optional) The type of the dataset (e.g. 'caltech', 'synthetic')
            bucket (str): (optional) An S3 bucket to store the dataset in
            prefix (str): (optional) An S3 prefix directory to store dataset objects in, within the bucket
            region (str): (optional) Region to place/use dataset from
        """
        self.name = name
        self.format = format
        self.type = type
        self.bucket_name = bucket
        self.region = region
        self.prefix = prefix or 'datasets'

    @abstractmethod
    def build(self, overwrite=False):
        pass

    @property
    def s3_path(self):
        """Return the path part of the S3 URI of this dataset."""
        return "/{}/{}".format(self.prefix.rstrip('/'), self.name)

    @property
    def s3_uri(self):
        """Return the S3 URI of this dataset."""
        return "s3://{}/{}/{}".format(self.bucket_name, self.prefix.rstrip('/'), self.name)

    def __str__(self):
        """Return the name of this dataset."""
        return self.name

    def _exists(self):
        s3 = utils.get_s3_resource()
        bucket = s3.Bucket(self.bucket_name)
        try:
            for _ in bucket.objects.filter(Prefix=self.s3_path.lstrip('/')):
                return True
            return False
        except Exception as e:
            print(e)
            return False


class BenchmarkDataset(Dataset):
    def __init__(self, name, **kw):
        """Base constructor for BenchmarkDataset."""
        super().__init__(name, **kw)

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _write_image_raw(file, arr, label):
        """Write a raw image from numpy-array without compression"""
        height, width, depth = arr.shape
        feature = {
            'image': BenchmarkDataset._bytes_feature(arr.tobytes()),
            'label': BenchmarkDataset._int_feature(label),
            'height': BenchmarkDataset._int_feature(height),
            'width': BenchmarkDataset._int_feature(width),
            'depth': BenchmarkDataset._int_feature(depth)}
        BenchmarkDataset._write_example(file, feature)

    @staticmethod
    def _write_image_bytes(file, bytes, label):
        """Write image directly from byte stream (which may have been encoded)"""
        feature = {
            'image': BenchmarkDataset._bytes_feature(bytes),
            'label': BenchmarkDataset._int_feature(label)}
        BenchmarkDataset._write_example(file, feature)

    @staticmethod
    def _write_example(file, feature):
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        file.write(example.SerializeToString())       
