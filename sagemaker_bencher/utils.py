# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


import os
import time
from functools import lru_cache
from collections.abc import Mapping

import boto3
import botocore
from botocore.exceptions import ClientError


def list_to_dict(inp, prefix='t'):
    if isinstance(inp, dict):
        return inp
    elif isinstance(inp, list):
        return {f'{prefix}{i}':p for i,p in enumerate(inp)}
    else:
        raise NotImplementedError("Must be either list or dict, but '%s' provided.." % type(inp))
        

def get_files_from_dir(top_dir, file_extensions=None, sort=False, filters=None):
    walker = sorted(os.walk(top_dir)) if sort else os.walk(top_dir)

    for root, _, files in walker:
        for f in files:
            if file_extensions and not _check_ext(f, file_extensions):
                continue
            if filters and any([func(f) for func in filters]):
                continue
            yield os.path.join(root, f)


def get_bucket(bucket=None, region='eu-central-1', create_if_needed=True):
    """Return a bucket (create if needed) for storing SageMaker benchmarking data."""
    boto_session = get_boto3_session()
    s3 = boto_session.resource('s3')
    account = boto_session.client('sts').get_caller_identity()['Account']
    bucket = bucket or 'sagemaker-benchmark-{}-{}'.format(region, account)

    if not bucket_exists(bucket) and create_if_needed:
        print("Creating new bucket '{}' in region '{}'..".format(bucket, region))
        if region == 'us-east-1':   # https://github.com/boto/boto3/issues/125
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={'LocationConstraint': region})
        time.sleep(3)

    return bucket


def get_role_arn(role_name, region='eu-central-1'):
    """Return the arn for the role role_name."""
    iam = boto3.client('iam', region_name=region)
    retrieved_all_roles = False
    marker = None
    while not retrieved_all_roles:
        if marker:
            list_roles_response = iam.list_roles(Marker=marker)
        else:
            list_roles_response = iam.list_roles()
        marker = list_roles_response.get('Marker', None)
        retrieved_all_roles = (marker is None)
        for role in list_roles_response['Roles']:
            if role_name == role['RoleName']:
                return role['Arn']
    return None


def bucket_exists(bucket_name):
    s3 = get_s3_resource()
    try:
        s3.meta.client.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        # If a client error is thrown, then check that it was a 404 error.
        # If it was a 404 error, then the bucket does not exist.
        error_code = int(e.response['Error']['Code'])
        if error_code == 403:
            print("Private Bucket ('%s'). Forbidden Access!" % bucket_name)
            return True
        elif error_code == 404:
            return False


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def _check_ext(filename, file_extensions):
    _, ext = os.path.splitext(filename)
    if ext.lower() not in file_extensions:
        print("File extention of '%s' is not supported.." % filename)
        return False
    return True


def download_file_from_google_drive(file_id, root, filename=None):
    """Download a Google Drive file from  and place it in root.
    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
    """
    import gdown

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)
    gdown.download(f'https://drive.google.com/uc?id={file_id}', fpath, quiet=False)


@lru_cache()
def get_boto3_session(region=None):
    return boto3.Session(region_name=region)


@lru_cache()
def get_s3_resource(region=None):
    config = botocore.config.Config(max_pool_connections=os.cpu_count()*2)
    return get_boto3_session().resource('s3', config=config)
    

def upload_file(bucket_name, file, key):
    s3 = get_s3_resource()
    bucket = s3.Bucket(bucket_name)
    bucket.put_object(Key=key, Body=open(file, 'rb'))
