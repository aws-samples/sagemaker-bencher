# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


from sagemaker_bencher.dataset import Dataset


class S3PrefixDataset(Dataset):
    """A generic dataset that has been prepared and stored on S3."""

    def build(self):
        """Since the s3prefix has to be already pre-tbuilt, it merely checks that it already exists."""
        if self._exists():
            print(f"Dataset '{self.name}' found on '{self.s3_uri}'..")
        else:
            print(f"No dataset was found on '{self.s3_uri}'.. This dataset has to be manually built first!")
            raise FileNotFoundError(f"No dataset found on '{self.s3_uri}'..")

