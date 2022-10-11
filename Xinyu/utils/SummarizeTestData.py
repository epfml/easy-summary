# -- fix path --
from ast import keyword
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
from preprocessor import yield_lines, get_data_filepath, yield_sentence_pair,tokenize,TURKCORPUS_DATASET, WIKI_DOC, WIKI_DOC_Small,WIKI_DOC_MATCH, WIKI_DOC_CLEAN, WIKI_DOC_MATCH, D_WIKI, D_WIKI_SMALL, D_WIKI_CLEAN, D_WIKI_MATCH
from summarizer import Summarizer
import numpy as np
import re
from keybert import KeyBERT
from transformers.pipelines import pipeline

hf_model = pipeline("feature-extraction", model="distilbert-base-cased")

kw_model = KeyBERT(model = 'all-mpnet-base-v2')


# model = Summarizer(model='distilbert-base-uncased')
dataset = WIKI_DOC_MATCH

phases = ['train', 'valid', 'test' ]


for ps in phases:
    simple_file_path = get_data_filepath(dataset,ps, 'simple')
    complex_file_path = get_data_filepath(dataset,ps, 'complex')
    save_complex_summary_path = get_data_filepath(WIKI_DOC_MATCH,ps, 'complex_kw_num4_div0.7')
    #save_simple_summary_path = get_data_filepath(WIKI_DOC_MATCH,ps, 'simple')
    summary = []
    cnt=1
    comps = []
    simps = []
    for complex_sentence, simple_sentence in yield_sentence_pair(complex_file_path, simple_file_path):
        # L1 = len(tokenize(complex_sentence))
        # L2 = len(tokenize(simple_sentence))

        # if L1<L2-5:
        #     continue
        # comps.append(complex_sentence)
        # simps.append(simple_sentence)

        # x = complex_sentence
        # y = re.sub('\(.*?\)', '', x) 
        # y = re.sub(' +', ' ', y)
        # comps.append(y)

        # x = simple_sentence
        # y = re.sub('\(.*?\)', '', x) 
        # y = re.sub(' +', ' ', y)
        # simps.append(y)

        key_words = kw_model.extract_keywords(complex_sentence, keyphrase_ngram_range=(1, 1), stop_words=None, 
                                            top_n = 7, use_mmr = True,diversity = 0.7)
        added = ''
        
        for i in range(min(4, len(key_words))):
            #print(len(key_words[i]))
            #added = key_words[i][0]+ ' ' + added
            #added = key_words[i][0] + '_' + str(round(key_words[i][1],2)) + ' ' + added
            complex_sentence = key_words[i][0] + "_" + str(round(key_words[i][1],2)) + ' ' + complex_sentence
            
        #complex_sentence = added + '</s> ' +complex_sentence
        summary.append(complex_sentence)

        cnt+=1
        print(cnt)
        
    
    file_write_obj = open(save_complex_summary_path, 'w', encoding='utf-8')
    for var in summary:
        file_write_obj.write(var)
        file_write_obj.write('\n')
    file_write_obj.close()

    # file_write_obj = open(save_simple_summary_path, 'w', encoding='utf-8')
    # for var in simps:
    #     file_write_obj.write(var)
    #     file_write_obj.write('\n')
    # file_write_obj.close()

    print("done")

