# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
from preprocessor import yield_lines, get_data_filepath, yield_sentence_pair,tokenize,TURKCORPUS_DATASET, WIKI_DOC, WIKI_DOC_Small,D_WIKI, D_WIKI_SMALL
from summarizer import Summarizer
import numpy as np
from keybert import KeyBERT
from transformers.pipelines import pipeline

hf_model = pipeline("feature-extraction", model="distilbert-base-cased")

kw_model = KeyBERT(model = 'all-mpnet-base-v2')


# model = Summarizer(model='distilbert-base-uncased')
dataset = D_WIKI

phases = ['train', 'valid']


for ps in phases:
    simple_file_path = get_data_filepath(dataset,ps, 'simple')
    complex_file_path = get_data_filepath(dataset,ps, 'complex')
    save_complex_summary_path = get_data_filepath(dataset,ps, 'complex_keynumC')
    summary = []
    cnt=1
    for complex_sentence, simple_sentence in yield_sentence_pair(complex_file_path, simple_file_path):
        key_words = kw_model.extract_keywords(complex_sentence, keyphrase_ngram_range=(1, 1), stop_words=None, top_n = 7, diversity = 0.5)
        added = ''
        
        for i in range(min(3, len(key_words))):
            #print(len(key_words[i]))
            #added = added + key_words[i][0] + ' </s> '
            complex_sentence = key_words[i][0] + "_" + str(round(key_words[i][1],2)) + ' ' + complex_sentence
        
        #complex_sentence = added+complex_sentence
        summary.append(complex_sentence)
        cnt+=1
        print(cnt)
        
    
    file_write_obj = open(save_complex_summary_path, 'w', encoding='utf-8')
    for var in summary:
        file_write_obj.write(var)
        file_write_obj.write('\n')
    file_write_obj.close()
    print("done")

