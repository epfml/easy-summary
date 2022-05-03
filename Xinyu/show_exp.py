from preprocessor import RESOURCES_DIR,DATASETS_DIR,PROCESSED_DATA_DIR,DUMPS_DIR,dump,load_dump
import pathlib
from Ts_T5 import T5FineTuner
from argparse import ArgumentParser
from preprocessor import Preprocessor, WIKILARGE_FILTER_DATASET,load_preprocessor
import pandas as pd
def parse_arguments1():
    p = ArgumentParser()
                  
    p.add_argument('-t', '--trials', type=int, default=5,
                  help='number of trials for hyperparameter search')
    p.add_argument('--seed', type=int, default=42, help='randomization seed')
    p.add_argument('--features_kwargs', default= {
    # 'WordRatioFeature': {'target_ratio': 0.8},
    'CharRatioFeature': {'target_ratio': 0.96},
    'LevenshteinRatioFeature': {'target_ratio': 0.75},
    'WordRankRatioFeature': {'target_ratio': 0.75},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.76}
})
    p = T5FineTuner.add_model_specific_args(p)
    #p = pl.Trainer.add_argparse_args(p)
    args,_ = p.parse_known_args()
    return args
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

checkpoint_file = 'experiments/exp_wiki_paragh/checkpoint-epoch=0.ckpt'

'''
pickle2: 0.95 0.75 0.75 0.75 
pickle3: 0.95 0.8 0.7 0.75
'''

def save_preprocessor1(preprocessor):
    DUMPS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_DUMP_FILE = DUMPS_DIR / 'preprocessor.pickle3'
    dump(preprocessor, PREPROCESSOR_DUMP_FILE)


def load_preprocessor1():
    PREPROCESSOR_DUMP_FILE = DUMPS_DIR / 'preprocessor.pickle3'
    if PREPROCESSOR_DUMP_FILE.exists():
        return load_dump(PREPROCESSOR_DUMP_FILE)
    else:
        return None


args = parse_arguments1()
print(args.features_kwargs)
preprocessor = Preprocessor(args.features_kwargs)
#print(preprocessor.encode_sentence('This is a test sentence.'))
save_preprocessor1(preprocessor)
# T5FineTuner.preprocessor = load_preprocessor1()
# model = T5FineTuner(args).load_from_checkpoint(checkpoint_file)


# wiki_large_valid_complex = pd.read_csv('resources/datasets/wiki_paraghF/wiki_paraghF.valid.complex', sep = '\t',header = None)
# wiki_large_valid_simple = pd.read_csv('resources/datasets/wiki_paraghF/wiki_paraghF.valid.simple', sep='\t',header=None)

# rand_loc = 310
# tmp = wiki_large_valid_complex.iloc[rand_loc,0]
# #model's generation
# print("Generation: ",model.generate(tmp))
# #complex
# print("Complex: ",tmp)
# #simple sentence
# print("Simple: ", wiki_large_valid_simple.iloc[rand_loc,0])