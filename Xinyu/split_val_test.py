from pathlib import Path
import sys
# sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessor import RESOURCES_DIR, get_data_filepath, WIKI_DOC, DATASETS_DIR



filepath = get_data_filepath(WIKI_DOC, 'valid', 'complex')
targetpath = get_data_filepath(WIKI_DOC, 'valid','simple')

with Path(filepath).open('r') as f1, Path(targetpath).open('r') as f2:
    complex_sents = f1.read().split('\n')
    simple_sents = f2.read().split('\n')

    complex_sents = np.array(complex_sents)
    simple_sents = np.array(simple_sents)

    x_val, x_test, y_val, y_test = train_test_split(complex_sents, simple_sents, test_size=0.03, random_state=42)

    #### save to file using pandas in txt format
    #x_val = pd.DataFrame(x_val)
    x_test = pd.DataFrame(x_test)
    #y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)

    #x_val.to_csv(get_data_filepath(WIKI_DOC, 'val', 'complex'),sep = '\t', index=False, header=False, encoding='utf-8')
    x_test.to_csv(get_data_filepath(WIKI_DOC, 'valid_small', 'complex'),sep = '\t', index=False, header=False, encoding='utf-8')
    #y_val.to_csv(get_data_filepath(WIKI_DOC, 'val', 'simple'),sep = '\t', index=False, header=False, encoding='utf-8')
    y_test.to_csv(get_data_filepath(WIKI_DOC, 'valid_small', 'simple'),sep = '\t', index=False, header=False, encoding='utf-8')
    






