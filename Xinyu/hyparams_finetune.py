# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
# -- end fix path --
import Preprocessor
from main import log_params, log_stdout, run_training
from evaluate import evaluate_on_WIKIDOC
from Preprocessor import get_experiment_dir, WIKI_DOC, EXP_DIR
from optparse import OptionParse
import time
import optuna

def run_train_tuning(trail, args, dataset):
    dir_name = f'{int(time.time() * 1000000)}'
    exp_dir_path = EXP_DIR / f'tuning_experiments/exp_{dir_name}'
    exp_dir_path.mkdir(parents=True, exist_ok=True)
    args.output_dir = exp_dir_path

    log_params(args.output_dir / "params.json", vars(args))

    preprocessor = Preprocessor(args.features_kwargs)
    preprocessor.preprocess_dataset(dataset)

    args.dataset = dataset
    print("Dataset: ", args.dataset)

    run_training(trail, args, dataset)

def evaluate(params):
    features_kwargs = {
        'WordRatioFeature': {'target_ratio': params['WordRatio']},
        'CharRatioFeature': {'target_ratio': params['CharRatio']},
        'LevenshteinRatioFeature': {'target_ratio': params['LevenshteinRatio']},
        'WordRankRatioFeature': {'target_ratio': params['WordRankRatio']},
        'DependencyTreeDepthRatioFeature': {'target_ratio': params['DepthTreeRatio']}
    }
    








