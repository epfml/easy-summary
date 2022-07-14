from preprocessor import RESOURCES_DIR,DATASETS_DIR,PROCESSED_DATA_DIR,DUMPS_DIR,dump,load_dump
from argparse import ArgumentParser
from preprocessor import Preprocessor
import pandas as pd
def parse_arguments1():
    p = ArgumentParser()
    p.add_argument('--features_kwargs', default= {
    # 'WordRatioFeature': {'target_ratio': 0.8},
    'CharRatioFeature': {'target_ratio': 0.96},
    'LevenshteinRatioFeature': {'target_ratio': 0.75},
    'WordRankRatioFeature': {'target_ratio': 0.75},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.76}
})
    args,_ = p.parse_known_args()
    return args

'''
pickle2: 0.95 0.75 0.75 0.75 
pickle3: 0.96 0.75 0.75 0.76
'''

def save_preprocessor1(preprocessor):
    DUMPS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_DUMP_FILE = DUMPS_DIR / 'preprocessor.pickle3'
    dump(preprocessor, PREPROCESSOR_DUMP_FILE)


args = parse_arguments1()
print(args.features_kwargs)
preprocessor = Preprocessor(args.features_kwargs)
save_preprocessor1(preprocessor)
