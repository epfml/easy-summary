'''
Main Program: 
> python main.py
'''
# -- fix path --

import torch
#torch.multiprocessing.set_start_method('forkserver', force=True)
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
# -- end fix path --

from preprocessor import D_WIKI,D_WIKI_SMALL,D_WIKI_CLEAN, D_WIKI_MATCH, WIKI_DOC_MID, TURKCORPUS_DATASET, EXP_DIR, WIKI_DOC, WIKI_PARAGH_SMALL, WIKI_DOC_Small, WIKI_PARA_DATASET, Preprocessor,EPFL_NEWS, WIKILARGE_DATASET,WIKILARGE_FILTER_DATASET,WIKI_PARAGH_FILTER_DATASET, WIKI_DOC_CLEAN, WIKI_DOC_MATCH
import time
import json

#from contextlib import contextmanager
import argparse
#from Ts_T5 import T5FineTuner, train
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback
#from new_model import SumSim, train
#from Ts_BART import BartFineTuner, train
from T5_2 import SumSim, train
#from Bart2 import SumSim, train
#from Bart_baseline_finetuned import BartBaseLineFineTuned, train
#from T5_baseline_finetuned import T5BaseLineFineTuned, train


def parse_arguments():
    p = ArgumentParser()
                  
    p.add_argument('-t', '--trials', type=int, default=5,
                  help='number of trials for hyperparameter search')
    p.add_argument('--seed', type=int, default=42, help='randomization seed')
#     p.add_argument('--features_kwargs', default= {
#     # 'WordRatioFeature': {'target_ratio': 0.8},
#     'CharRatioFeature': {'target_ratio': 0.8},
#     'LevenshteinRatioFeature': {'target_ratio': 0.8},
#     'WordRankRatioFeature': {'target_ratio': 0.8},
#     'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
# })
    #p = T5FineTuner.add_model_specific_args(p)
    p = SumSim.add_model_specific_args(p)
    #p = BartFineTuner.add_model_specific_args(p)
    #p = BartBaseLineFineTuned.add_model_specific_args(p)
    #p = T5BaseLineFineTuned.add_model_specific_args(p)
    p = pl.Trainer.add_argparse_args(p)
    args,_ = p.parse_known_args()
    return args

# class MetricsCallback(pl.Callback):
#   def __init__(self):
#     super().__init__()
#     self.metrics = []
  
#   def on_validation_end(self, trainer, pl_module):
#       self.metrics.append(trainer.callback_metrics)

# Create experiment directory
def get_experiment_dir(create_dir=False):
    dir_name = f'{int(time.time() * 1000000)}'
    path = EXP_DIR / f'exp_{dir_name}'
    if create_dir == True: path.mkdir(parents=True, exist_ok=True)
    return path

def log_params(filepath, kwargs):
    filepath = Path(filepath)
    kwargs_str = dict()
    for key in kwargs:
        kwargs_str[key] = str(kwargs[key])
    json.dump(kwargs_str, filepath.open('w'), indent=4)

# @contextmanager
# def log_stdout(filepath, mute_stdout=False):
#     '''Context manager to write both to stdout and to a file'''

#     class MultipleStreamsWriter:
#         def __init__(self, streams):
#             self.streams = streams

#         def write(self, message):
#             for stream in self.streams:
#                 stream.write(message)

#         def flush(self):
#             for stream in self.streams:
#                 stream.flush()

#     save_stdout = sys.stdout
#     log_file = open(filepath, 'w')
#     if mute_stdout:
#         sys.stdout = MultipleStreamsWriter([log_file])  # Write to file only
#     else:
#         sys.stdout = MultipleStreamsWriter([save_stdout, log_file])  # Write to both stdout and file
#     try:
#         yield
#     finally:
#         sys.stdout = save_stdout
#         log_file.close()




def run_training(args, dataset):

    args.output_dir = get_experiment_dir(create_dir=True)
    # logging the args
    log_params(args.output_dir / "params.json", vars(args))

    # if without tokens, delete it
    # preprocessor = Preprocessor(args.features_kwargs)
    # preprocessor.preprocess_dataset(dataset)
    # dataset 
    args.dataset = dataset
    print("Dataset: ",args.dataset)
    train(args)

    #### add logging process if you want
    # with log_stdout(args.output_dir / "logs.txt"):
    #     #train_summary(args)
    #     train(args)



     ## MLO98 (tmux 1): T5_2 D_wiki 20Sim+1Sum-10CosSim(ReLU) 384Hidden_size (on epoch3)
     ## MLO95 (tmux 0): T5_2 D_Wiki (whole) 20Sim+1Sum+15KL 512_SeqDim (on epoch3)


    ## T5_2 D-wiki_match keyword_num3_div0.5 0.53460 -> 0.52810 -> 0.52764 -> 0.52101 -> 0.51956 -> 0.51941 -> 0.51896
    ## T5_2 D_wiki-match 1Sim-0.5CosSim(ReLU)(H_sim*W, H2*W)0.54373 -> 0.52729 -> 0.51991 -> 0.51864
    ## T5_2 D_wiki_match 1Sim-0.01CosSim(ReLU)(H_sim*W, H2*W) 0.53203 -> 0.52532 -> 0.52503 -> 0.51998 -> 0.51843 -> 0.51641
    ## T5_2 D_wiki_match 1Sim-0.1CosSim(ReLU)(H_sim*W, H2*W) 0.53762 -> 0.52782 -> 0.52212
    ## T5_2 D_wiki_match 1Sim-1CosSim(ReLU)(H_sim*W, H2*W) 0.57321 -> 0.52840 -> 0.52626 -> 0.52005
    ## T5_2 D-wiki-match kw3_div0.7_sep 0.53773 -> 0.52873 -> 0.52608 -> 0.52495 -> 0.52211 -> 0.52148
    ## T5_2 D_wiki_match kw_num3_div0.7 (label smoothing 0.1) 0.53183 -> 0.52506 -> 0.52374 -> 0.51900 -> 0.51856 -> 0.51732
    ## T5_2 D-wiki-match kw_num3_div0.7 0.52716 -> 0.52635 -> 0.52135 -> 0.52082 -> 0.51680 -> 0.51622 -> 0.51512 -> 0.51467
    ## T5_2 D_wiki_match kw_num4_div0.7 0.52805 -> 0.52301 -> 0.52188 -> 0.51986 -> 0.51751 -> 0.51531
    ## T5_2 D_wiki_match kw_num4_div0.9 0.53169 -> 0.52616 -> 0.52415 -> 0.51839 -> 0.51835 -> 0.51590
    ## T5_2 D-wiki-match kw_num3_div0.9 0.53095 -> 0.52483 -> 0.52011 -> 0.51872 -> 0.51796 -> 0.51628
    ## T5_2 D-wiki-match kw_num3_div0.9 (label smoothing 0.1) 0.53295 -> 0.52955 -> 0.52592 -> 0.52094 -> 0.51802
    ## T5_2 D_wiki_match 1Sim-0.5CosSim(H_sim, H2) 0.65300
    ## T5_2 D_wiki_match 1Sim+1Sum 0.57021 -> 0.56645



##STOP- MLO97 (tmux 0): T5_2 D_wiki_match original loss 0.52277 -> 0.52057
##STOP- MLO100 (tmux 0): T5_2 D_wiki_match 1Sim+0.1Sum 0.57238
##STOP- MLO95 (tmux 1): T5_2 Wiki_Doc_match kw_num4_div0.7
##STOP- MLO100 (tmux 0): T5_2 D_wiki_match 1Sim+0.1Sum 0.57238
##STOP- MLO96 (tmux 0): T5_2 D_wiki_match 1Sim+0.01Sum 0.58517
## MLO94 (tmux 1): T5_2 D_wiki_match 1Sim-2CosSim(ReLU)(H_sim*W, H2*W) 0.52839 -> 0.52745 -> 0.52084 -> 0.52016
##STOP- MLO100 (tmux 0): T5_2 D_wiki_match 1Sim+0.001Sum 0.56481
##STOP- MLO99 (tmux 0): T5_2 D_wiki_match (1st stage freeze) original loss 0.53515 -> 0.52706 -> 0.52037 -> 0.51967 -> 0.51731
## MLO96 (tmux 1): T5_2 D_Wiki_match 1Sim-0.001CosSim(ReLU)(H_sim*W, H2*W) 0.53350 -> 0.52613
##STOP- MLO98 (tmux 0): continue T5_2 D_wiki_match original loss 0.51914 -> 0.51888 -> 0.51751
##STOP- MLO95 (tmux 0): continue T5_2 D_wiki_match kw_num3_div0.7 0.51502
## MLO94 (tmux 0): T5_2 wiki_doc_match 1Sim-0.01CosSim(ReLU)(H_sim*W, H2*W) 0.50801
## MLO95 (tmux 0): T5_2 wiki_doc_match 1Sim-0.1CosSim(ReLU)(H_sim*W, H2*W) 0.51155 -> 0.51122
## MLO98 (tmux 0): T5_2 wiki_doc_match 1Sim-0.5CosSim(ReLU)(H_sim*W, H2*W) 0.53176 -> 0.53067
    ## MLO96 (tmux 1): T5_2 D_wiki-match 1Sim-1CosSim(ReLU)(H_sim, H2) 0.61288
dataset = WIKI_DOC_MATCH

args = parse_arguments()
run_training(args, dataset)

##########################
### fine-tuning
# dataset = EPFL_DATASET
# args.output_dir = get_experiment_dir(create_dir=True)
# log_params(args.output_dir / "params.json", vars(args))

# preprocessor = Preprocessor(args.features_kwargs)
# preprocessor.preprocess_dataset(dataset)
# args.dataset = dataset
# model = T5FineTuner.load_from_checkpoint('/home/xinyzhou/text_sum_sim/experiments/exp_1647193991142787/checkpoint-epoch=1.ckpt')

# def optuna_obj(trial, args, model):

#     args.learning_rate = trial.suggest_categorical('lr',[1e-6,5e-6,1e-5])
#     args.weight_decay = trial.suggest_categorical('weight_decay',[0.005,0.01,0.05])
#     metrics_callback = MetricsCallback()
#     cp_callback = pl.callbacks.ModelCheckpoint(
#       dirpath=args.output_dir,
#         filename="checkpoint-{epoch}",
#         monitor="val_loss",
#         verbose=True,
#         mode="min",
#         save_top_k=1
#     )
#     pr_callback = PyTorchLightningPruningCallback(trial,monitor = 'val_loss')
#     model = model

#     # train_params = dict(
#     #       accumulate_grad_batches=args.gradient_accumulation_steps,
#     #       gpus=args.n_gpu,
#     #       max_epochs=args.num_train_epochs,
#     #       # early_stop_callback=False,
#     #       # gradient_clip_val=args.max_grad_norm,
#     #       # checkpoint_callback=checkpoint_callback,
#     #       callbacks=[pr_callback, cp_callback,
#     #                  metrics_callback],
#     #       # logger=TensorBoardLogger(f'{args.output_dir}/logs'),
#     #       num_sanity_val_steps=args.nb_sanity_val_steps,  # skip sanity check to save time for debugging purpose
#     #       # plugins='ddp_sharded',
#     #       #progress_bar_refresh_rate=1,

#     # )
#     early_stopping = EarlyStopping(
#       monitor='val_loss',
#       patience=6
#     )
#     trainer = pl.Trainer.from_argparse_args(
#       args,
#       callbacks=[pr_callback, cp_callback,
#                  metrics_callback, early_stopping],
#       num_sanity_val_steps=args.nb_sanity_val_steps
#     )

#     trainer.fit(model)

#   #trainer.test(model)
#     return min([x['val_loss'].item() for x in metrics_callback.metrics])

# pruner = optuna.pruners.MedianPruner()
# study = optuna.create_study(direction='maximize',pruner=pruner)
# study.optimize(lambda x: optuna_obj(x, args, model),n_trials=args.trials)

