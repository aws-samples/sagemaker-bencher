import sagemaker
import time
from sagemaker_bencher import experiment


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help="YAML-file with experiment definitions")
    args = parser.parse_args()

    print("SageMaker SDK:", sagemaker.__version__)

    experiment = experiment.Experiment.from_file(args.file)

    tic = time.time()

    experiment.start()

    print(f"Experiment finished in: {time.time() - tic} sec..")
