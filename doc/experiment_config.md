## Experiment configuration file
Let's discuss the structure of experimentation config on the example of `experiments/blog-benchmarks-all.yml`.


### Main experiment settings
The experiment config begins by setting a couple of essential experiment parameters:

- *name* -- defines the name of the experiment under which it will be tracked in SageMaker Experiments. The name must be unique within an account.
- *description* -- a string that describes the experiments.
- *role* -- the name of AWS IAM role that will be used by Amazon SageMaker training jobs to, for example, access training data and model artifacts.
- *region* -- the name of the AWS region for experiment jobs and other artifacts.
- *bucket* -- an S3 bucket name where to store experiment artifacts in. Set to `null` to use a default bucket name (recommended).
- *output_prefix* -- an S3 bucket prefix to store experiment artifacts in.
- *parallelism* -- defines how many SageMaker training jobs will be run concurrently at any point of time. Set this up accordingly to stay within the [SageMaker Training quotas](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html) (note that you can also [request SageMaker quota increase](https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html) as needed). Set to `0` to disable concurrent execution, and run experiment sequantially (e.g., for debugging).
- *repeat* -- specifies how many times to repeat each trial.
- *disable_profiler* -- specifies whether [SageMaker Debugger monitoring and profiling](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html) will be enabled or disabled.
- *order* -- can be either `default`, `interleave`, or `random`. Specifies the order to run the (`repeat`ed) trials (e.g., to reduce resource congestion).
- *clean* -- set to `True` to automatically delete all previous trials and start this experiment from a clean slate. Set to `False` in order to append new trials to an existing experiment.

Excerpt from `blog-benchmarks-all.yml`:
```
name: blog-benchmarks-all
description: 'Benchmarks -- Blog: Choose the best data source for your Amazon SageMaker training job'
role: SageMakerRoleBenchmark
region: us-east-1
bucket: null
output_prefix: experiments
parallelism: 6
repeat: 3
disable_profiler: True
order: interleave
clean: True
```

### Dataset definitions
The dataset is defined under `datasets` first-level key of the experiment config file and can either be a `*.yml`-file itself with dataset definitions (that will be parsed upon experiment initialization), or a dictionary where every key is the *name* of the dataset, and the value is another dictionary with *dataset properties*. The *dataset properties* indicate the *type* of the dataset and where to find it. If the dataset is not found at the specified S3 bucket/prefic location, and its *type* is either `synthetic` or `caltech` (you can also implement your own new dataset type), then the dataset will either be automatically downloaded (in case of `caltech`), or synthesized locally (in case of `synthetic`), and then prepared according to the other *dateset properties* before being uploaded to the specified S3 bucket/prefix. The benchmark experiment will start automatically afterwards. That is, building of the dataset is normally required only once.

Currently supported dataset types (specified by `type` parameter in the *dataset properties*) are `caltech`, `synthetic`, and `s3prefix`. The latter (`s3prefix`) allows to use any custom dataset the has been already stored on S3 -- in this case one needs to provide S3 `bucket` name, as well as S3 `prefix` where the dataset is stored.

For example, the snippet below defines two datasets with names `MyTrainDatasetSplit` and `MyTestDatasetSplit` that are stored under `s3://sagemaker-benchmark-us-east-1-24271126XXXX/datasets/MyTrainDatasetSplit` and `s3://sagemaker-benchmark-us-east-1-24271126XXXX/datasets/MyTestDatasetSplit`, respectively.

```
datasets:
  MyTrainDatasetSplit:
    type: s3prefix
    bucket: sagemaker-benchmark-us-east-1-24271126XXXX
    prefix: 'datasets'

  MyTestDatasetSplit:
    type: s3prefix
    bucket: sagemaker-benchmark-us-east-1-24271126XXXX
    prefix: 'datasets'
```

### Trial definitions
A *base trial* specifies parameters that by default apply to all other trials, *unless overridden or extended in the trial definition later*. In other words, the specifications of *each single trial* inherit from the *base trial*, but each single trial can *override* or *add any extra parameters* to the settings specified in base trial.

For example, this excerpt from `blog-benchmarks-all.yml` below defines 3 trials with all but two identical parameters (`dataset` and `input_mode`). The rest of the parameters are inherited from *base trial*, and are thus identical.

```
base_trial:
  script: script-model.py
  source_dir: scripts
  framework: tf
  framework_version: '2.4.1'
  py_version: py37
  volume_size: 500
  instance_type: ml.p3.2xlarge
  instance_count: 1
  max_run: 36000
  inputs:
    train:
      dataset: <to be overriden>
      input_mode: <to be overriden>
  hyperparameters:
    key1: value1
    key2: value2
    (...)

trials:
  - inputs:
      train:
        dataset: Caltech-jpg-1x
        input_mode: file
    
  - inputs:
      train:
        dataset: Caltech-jpg-1x
        input_mode: fsx
    
  - inputs:
      train:
        dataset: Caltech-jpg-1x
        input_mode: ffm

  (...)
```

The base trial in this case defines several parameters for the trials:

- *script* -- the name of the local training script that will be executed as the entry point to SageMaker training job. If `source_dir` is specified, then `script` must point to a file located at the root of `source_dir`.
- *source_dir* -- the path to a directory entry point script and any other dependencies.
- *framework* -- can be either `tf` for TensorFlow, `pt` for PyTorch, or `mx` gfor MXNet.
- *framework_version* -- version of the framewort you want to use for executing your model training code.
- *py_version* -- Python version you want to use for executing your model training code.
- *volume_size* -- size in GB of the EBS volume to use for storing input data during training. Must be large enough to store training data if File mode is used.
- *instance_type* -- Type of EC2 instance to use for training, for example, ‘ml.p3.2xlarge’.
- *instance_count* -- Number of Amazon EC2 instances to use for training.
- *max_run* -- Timeout in seconds for training (default: 24 * 60 * 60). After this amount of time Amazon SageMaker terminates the job regardless of its current status.
- *hyperparameters* -- a dictionary with any custom hyperparameters that will be passed to the training script.
- *inputs* -- a dictionary, where each key is the name of the input channel name, and the value is a a dictionary with settings for that input channel. For example, the snippet below defines two input channels (named `train` and `test`), which will read `MyTrainDatasetSplit` and `MyTestDatasetSplit`, respectively (defined earlier under `datasets`) using Fast File Mode (`ffm`) in both cases. Accepted native SageMaker input modes: `file`, `pipe`, `ffm` and `fsx`.
```
inputs:
  train:
    dataset: MyTrainDatasetSplit
    input_mode: ffm
  test:
    dataset: MyTestDatasetSplit
    input_mode: ffm
```
