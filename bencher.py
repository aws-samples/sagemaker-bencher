import sagemaker
import time
from sagemaker_bencher import experiment


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help="YAML-file with experiment definitions")
    parser.add_argument('--bootstrap', action='store_true', help="Build all datasets without running benchmarks")
    parser.add_argument('--local', action='store_true', help="Run all benchmarks in SM local mode")
    args = parser.parse_args()

    print("SageMaker SDK:", sagemaker.__version__)

    experiment = experiment.Experiment.from_file(args.file)

    tic = time.time()

    experiment.start(bootstrap=args.bootstrap, local=args.local)

    print(f"Experiment finished in: {time.time() - tic} sec..")
