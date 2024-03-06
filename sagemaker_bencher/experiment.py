# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


"""
Implementation of the main benchmark experiment dispatcher.

The Experiment class implements the logic to run a series of Trials
as a part of a single benchmarking Experiment. An Experiment definition
can be read directly from a YAML-file, which fully defines all the
global Experiment settings, as well as the particular Trial parameters.

All Experiment/Trial runs and their outcomes are fully tracked with
SageMaker Experiments feature for versioning and futher analysis.

This file can be used in script mode with, e.g.:
`python experiment.py -f experiments/experiment_iopipe.yml`
"""

import concurrent.futures
import copy
import datetime
import random
import time

import boto3
import sagemaker
import smexperiments as sme
import smexperiments.experiment
import yaml
from sagemaker.debugger import FrameworkProfile, ProfilerConfig
from sagemaker.mxnet import MXNet
from sagemaker.pytorch import PyTorch
from sagemaker.tensorflow import TensorFlow

import sagemaker_bencher.utils as utils
from sagemaker_bencher.dataset import CaltechBenchmarkDataset, SyntheticBenchmarkDataset, S3PrefixDataset



class Experiment:

    sm_frameworks = {
        'tf': TensorFlow,
        'pt': PyTorch,
        'mx': MXNet
    }
    
    sm_input_modes = {
        'pipe': 'Pipe',
        'file': 'File',
        'ffm': 'FastFile',
    }

    def __init__(self,
                 name,
                 role,
                 region,
                 output_prefix, 
                 datasets,
                 base_trial,
                 trials,
                 bucket=None,
                 description="No description",
                 parallelism=8,
                 repeat=1,
                 order='default',
                 fsx=None,
                 disable_profiler=False,
                 clean=False):

        self.name = name
        self.role = role
        self.region = region
        self.output_prefix = output_prefix
        self.bucket = utils.get_bucket(bucket=bucket, region=region)
        self.description = description
        self.parallelism = parallelism
        self.repeat = repeat
        self.order = order
        self.fsx = fsx
        self.disable_profiler = disable_profiler
        self.clean = clean

        self._init_trials(base_trial, trials)
        self._init_datasets(datasets)

    def _init_trials(self, base_trial, trials):
        """Init all trial definitions"""
        self._trials = {} 
        trials = utils.list_to_dict(trials)
        for trial_name, trial_cfg in trials.items():
            self._trials[trial_name] = self.get_trial(base_trial, trial_cfg)

    def _init_datasets(self, datasets):
        """We need to instantiate datasets that are *only* used in trials"""
        self._datasets = {}     
        dataset_cfg = self.read_dataset_config(datasets)
        for trial_cfg in self._trials.values():
            for channel_cfg in trial_cfg['inputs'].values():
                dataset_name = channel_cfg['dataset']
                self._datasets[dataset_name] = self.get_dataset(dataset_name, dataset_cfg)
    
    @staticmethod
    def read_dataset_config(datasets):
        if isinstance(datasets, dict):
            return datasets
        elif isinstance(datasets, str) and datasets.endswith(('.yml', '.yaml')):
            print(f"Loading dataset definitions from '{datasets}'..")
            with open(datasets) as f:
                return yaml.safe_load(f)
        else:
            raise TypeError("Datasets attribute of wrong type!..")

    @classmethod
    def from_file(cls, experiment_file):
        with open(experiment_file) as f:
            cfg = yaml.safe_load(f)
        return cls(name=cfg['name'],
                   description=cfg.get('description'),
                   role=cfg['role'],
                   region=cfg['region'],
                   bucket=cfg.get('bucket'),
                   output_prefix=cfg['output_prefix'],
                   datasets=cfg['datasets'],
                   parallelism=cfg['parallelism'],
                   base_trial=cfg['base_trial'],
                   trials=cfg['trials'],
                   repeat=cfg.get('repeat', 1),
                   order=cfg.get('order', 'default'),
                   fsx=cfg.get('fsx'),
                   disable_profiler=cfg.get('disable_profiler', False),
                   clean=cfg.get('clean', False))

    @property
    def datasets(self):
        return self._datasets

    @property
    def trials(self):
        return self._trials

    @property
    def client(self):
        return boto3.client('sagemaker', region_name=self.region)

    def load_sm_experiment(self, clean=False):
        """Create a new or load an existing SM Experiment

        Args:
            clean (bool): delete all previous Trial records from
                          within the specified Experiment (if it existed)
        """
        try:
            sm_experiment = sme.experiment.Experiment.load(
                experiment_name=self.name,
                sagemaker_boto_client=self.client)
            if clean:
                print(f"Purging all previous trial records..")
                sm_experiment.delete_all(action='--force')
                time.sleep(5)
                sm_experiment = self.load_sm_experiment()
        except self.client.exceptions.ResourceNotFound as e:
            sm_experiment = sme.experiment.Experiment.create(
                experiment_name=self.name,
                description=self.description,
                sagemaker_boto_client=self.client)
        return sm_experiment

    def get_dataset(self, dataset_name, dataset_cfg):
        dataset_def = dataset_cfg[dataset_name].copy()
        dataset_type = dataset_def.get('type')
        if dataset_type == 'synthetic':
            return SyntheticBenchmarkDataset(dataset_name, region=self.region, **dataset_def)
        elif dataset_type == 'caltech':
            return CaltechBenchmarkDataset(dataset_name, region=self.region, **dataset_def)
        elif dataset_type == 's3prefix':
            return S3PrefixDataset(dataset_name, region=self.region, **dataset_def)
        else:
            raise NotImplementedError(f"Unknown dataset type '{dataset_def['type']}'..")
    
    def get_trial(self, base_trial, trial_cfg):
        """Merge the base and specific trial parameters and return them"""
        base_trial_copy = copy.deepcopy(base_trial)
        trial_def = utils.deep_update(base_trial_copy, trial_cfg)
        return trial_def

    def get_experiment_schedule(self):
        experiment_cross = [(t, r) for t in self.trials for r in range(self.repeat)]
        if self.order == 'interleave':
            experiment_cross.sort(key=lambda a: a[1])
        elif self.order == 'random':
            random.shuffle(experiment_cross)
        return experiment_cross

    def _set_channel_inputs_and_params(self, trial, job_name):
        inputs = {}
        config_extra_vars = {}
        env_vars = {'ML_FRAMEWORK': trial['framework']}
        
        # Set up env vars and inputs for all channels
        for ch_name, ch_cfg in trial['inputs'].items():
            dataset = self.datasets[ch_cfg['dataset']]
            env_vars[f'INPUT_MODE_CHANNEL_{ch_name.upper()}'] =  ch_cfg['input_mode']
            env_vars[f'DATASET_S3_URI_CHANNEL_{ch_name.upper()}'] =  dataset.s3_uri
            if hasattr(dataset, 'format'):
                env_vars[f'DATASET_FORMAT_CHANNEL_{ch_name.upper()}'] =  str(getattr(dataset, 'format'))
            if hasattr(dataset, 'num_classes'):
                env_vars[f'DATASET_NUM_CLASSES_CHANNEL_{ch_name.upper()}'] =  str(getattr(dataset, 'num_classes'))
            if hasattr(dataset, 'num_samples'):
                env_vars[f'DATASET_NUM_SAMPLES_CHANNEL_{ch_name.upper()}'] =  str(getattr(dataset, 'num_samples'))

            if ch_cfg['input_mode'] in ('file', 'ffm', 'pipe'):
                inputs[ch_name] = sagemaker.inputs.TrainingInput(
                    dataset.s3_uri,
                    s3_data_type='S3Prefix',
                    input_mode=self.sm_input_modes[ch_cfg['input_mode']])
            elif ch_cfg['input_mode'] == 'fsx':
                if self.fsx is None:
                    raise Exception(f"Job '{job_name}': FSx input mode is requested, "
                                      "but no FSx config was specified for this experiment..")
                config_extra_vars['subnets'] = self.fsx['subnets']
                config_extra_vars['security_group_ids'] = self.fsx['security_group_ids']
                inputs[ch_name] = sagemaker.inputs.FileSystemInput(
                    file_system_id=self.fsx['fsx_file_system_id'],
                    file_system_type='FSxLustre',
                    directory_path=self.fsx['fsx_file_system_directory_path'] + dataset.s3_path, # example: /fsx/datasets/caltech/Caltech-tfr-jpg-1x
                    file_system_access_mode='ro')
        return inputs, env_vars, config_extra_vars

    def _wait_and_finalize_job(self, trial, job_name, artifact_prefix=None):
        '''Wait for job to complete and finilize the logs, if successful. Returns elapsed job time.'''

        while True:
            job_desc = self.client.describe_training_job(TrainingJobName=job_name)
            job_status = job_desc['TrainingJobStatus'] 
            
            if job_status in ['Failed', 'Stopped']:
                raise Exception(f"!!! {job_status} job: '{job_name}'..")

            if job_status == 'Completed':
                job_times = {f"t_{d['Status'].lower()}": (d['EndTime'] - d['StartTime']).total_seconds()
                             for d in job_desc['SecondaryStatusTransitions']}
                job_times['t_training_sm'] = job_desc['TrainingTimeInSeconds']

                with sme.tracker.Tracker.load(
                    training_job_name=job_name,
                    artifact_bucket=self.bucket,
                    artifact_prefix=artifact_prefix,
                    sagemaker_boto_client=self.client) as trial_tracker:
                        for ch_name, ch_cfg in trial.pop('inputs').items():
                            dataset = self.datasets[ch_cfg['dataset']]
                            # add dataset format (if provided)
                            trial_tracker.log_parameter(f'input_format/{ch_name}', str(getattr(dataset, 'format', 'N/A')))
                            for k, v in ch_cfg.items():
                                trial_tracker.log_parameter(f'{k}/{ch_name}', v)
                        trial_tracker.log_parameters(trial)
                        trial_tracker.log_parameters(job_times)
                break
            else:
                time.sleep(30)

        return job_times['t_training_sm']

    def run_trial(self, trial, job_name):
        """Run single trial w/i the experiment"""

        role_arn = utils.get_role_arn(self.role, self.region)

        experiment_output = '{}/{}/{}'.format(
            self.bucket, self.output_prefix, self.name)
        code_location = 's3://' + experiment_output
        artifact_prefix = experiment_output + '/' + job_name
        output_path = code_location + '/' + job_name + '/output'

        hyperparams = trial.pop('hyperparameters')

        inputs, environment, channel_params = self._set_channel_inputs_and_params(trial, job_name)

        sm_trial = sme.trial.Trial.create(
            trial_name=job_name, 
            experiment_name=self.name,
            sagemaker_boto_client=self.client)

        sm_exp_config = {
            'ExperimentName': self.name, 
            'TrialName': sm_trial.trial_name,
            'TrialComponentDisplayName': 'Benchmark'}
        
        profiler_config = None if self.disable_profiler else ProfilerConfig(
                s3_output_path=code_location,
                system_monitor_interval_millis=100,
                framework_profile_params=FrameworkProfile())
        
        estimator_config = dict(
            entry_point=trial['script'],
            source_dir=trial['source_dir'],
            hyperparameters=hyperparams,
            role=role_arn,
            framework_version=trial['framework_version'],
            output_path=output_path,
            code_location=code_location,
            py_version=trial['py_version'],
            instance_count=trial['instance_count'],
            instance_type=trial['instance_type'],
            volume_size=trial['volume_size'],
            max_run=trial['max_run'],
            environment=environment,
            disable_profiler=self.disable_profiler,
            debugger_hook_config=False,
            profiler_config=profiler_config,
            sagemaker_session=sagemaker.Session(sagemaker_client=self.client))

        estimator_config.update(channel_params)   # Add any extra params (e.g. VPC settings if channels require VPC)
        inputs = inputs or None                   # Set explicitely to `None`, if no native SageMaker input modes are used

        SageMakerFramework = self.sm_frameworks[trial['framework']]
        model = SageMakerFramework(**estimator_config)
        model.fit(inputs, job_name=job_name, experiment_config=sm_exp_config, wait=(self.parallelism == 0))

        print(f"--> Submitted job: '{job_name}'..")
        job_time = self._wait_and_finalize_job(trial, job_name, artifact_prefix)
        return f"<-- Completed job: '{job_name}' (in {job_time} secs).."

    def start(self, bootstrap=False):
        if self.parallelism > 0:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.parallelism)
        else:
            executor = SyncExecutor()
        futures = []

        print(f"Building the datasets for '{self.name}' experiment..")
        for dataset in self.datasets.values():
            dataset.build()

        if bootstrap: 
            return
        
        print(f"Creating/loading SM Experiment with name '{self.name}'..")
        self.load_sm_experiment(clean=self.clean)

        experiment_schedule = self.get_experiment_schedule()

        print(f"Starting the '{self.name}' experiment with {len(experiment_schedule)} trials..")
        for i, (trial_name, trial_repeat) in enumerate(experiment_schedule, 1):
            trial = copy.deepcopy(self.trials[trial_name])
            job_name = '-'.join([
                self.name,
                f'{i}of{len(experiment_schedule)}',
                trial_name,
                str(trial_repeat), 
                datetime.datetime.now().strftime("%Y%m%d%H%M%S-%f")])
            future = executor.submit(self.run_trial, trial, job_name)
            futures.append(future)
            time.sleep(5)


        if self.parallelism > 0:
            for future in concurrent.futures.as_completed(futures):
                try:
                    benchmark_result = future.result()
                except Exception as e:
                    print(e)
                else:
                    print(benchmark_result)

                    
class SyncExecutor:
    def submit(self, run_trial, trial, job_name):
        run_trial(trial, job_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help="definition YAML-file")
    args = parser.parse_args()

    experiment = Experiment.from_file(args.file)
    experiment.start()
