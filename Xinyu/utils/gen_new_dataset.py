# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
from preprocessor import DATASETS_DIR,PROCESSED_DATA_DIR, D_WIKI_CLEAN, D_WIKI_MATCH, WIKI_DOC,WIKI_DOC_FILTER,WIKI_DOC_CLEAN,WIKI_PARAGH_FILTER_DATASET, WIKILARGE_DATASET,yield_sentence_pair,yield_lines, WIKI_PARA_DATASET, get_data_filepath, tokenize, write_lines
import numpy as np
from preprocessor import WIKILARGE_DATASET
from rouge import Rouge
import matplotlib.pyplot as plt


from keybert import KeyBERT
from transformers.pipelines import pipeline

hf_model = pipeline("feature-extraction", model="distilbert-base-cased")

kw_model = KeyBERT(model = 'all-mpnet-base-v2')

dataset = D_WIKI_CLEAN

TYPES = ['complex', 'simple']
PHASES = ['train', 'valid', 'test']

rouge = Rouge()
cnt = 0
# for phase in PHASES:
#     if phase == 'train':
#         complex_file = get_data_filepath(WIKILARGE_DATASET, phase, 'complex')
#         simple_file = get_data_filepath(WIKILARGE_DATASET, phase, 'simple')
#         for complex_sent, simple_sent in yield_sentence_pair(complex_file, simple_file):
#             rouge_score = rouge.get_scores(simple_sent, complex_sent)
#             if(rouge_score[0]['rouge-1']['f'] > 0.7):
#                 cnt+=1
#                 if cnt>5:
#                     print(simple_sent)
#                     print(complex_sent)
#                     print(rouge_score[0]['rouge-1']['f'])
#                     break
                
for ps in PHASES:
    save_complex_path = get_data_filepath(D_WIKI_MATCH, ps, 'complex')
    save_simple_path = get_data_filepath(D_WIKI_MATCH, ps, 'simple')
    simple_file_path = get_data_filepath(dataset,ps, 'simple')
    complex_file_path = get_data_filepath(dataset,ps, 'complex')
    cnt=0
    tot=0
    complex_lens = []
    simple_lens = []

    for complex_sentence, simple_sentence in yield_sentence_pair(complex_file_path, simple_file_path):
        sim_kw = kw_model.extract_keywords(simple_sentence, keyphrase_ngram_range=(1, 1), stop_words=None,top_n = 7, use_mmr = True,diversity = 0.5)
        com_kw = kw_model.extract_keywords(complex_sentence, keyphrase_ngram_range=(1, 1), stop_words=None, top_n = 7, use_mmr = True,diversity = 0.5)

        simlist = []


        for i in range(min(5, len(sim_kw))):
            simlist.append(sim_kw[i][0])
        simlist = set(simlist)

        for i in range(min(5, len(com_kw))):
            if com_kw[i][0] in simlist:
                #cnt+=1
                tot+=1
                print(tot)
                complex_lens.append(complex_sentence)
                simple_lens.append(simple_sentence)
                break
    
    file_write_obj = open(save_complex_path, 'w', encoding='utf-8')
    for var in complex_lens:
        file_write_obj.write(var)
        file_write_obj.write('\n')
    file_write_obj.close()

    file_write_obj = open(save_simple_path, 'w', encoding='utf-8')
    for var in simple_lens:
        file_write_obj.write(var)
        file_write_obj.write('\n')
    file_write_obj.close()

    print("done")


    #print(f'{ps}: {cnt}/{tot}')
# train: 23889/34727
# valid: 5892/8540
# test: 118/166


# for phase in PHASES:
#     complex_lens = []
#     simple_lens = []
#     complex_file = get_data_filepath(WIKI_DOC, phase, 'complex')
#     simple_file = get_data_filepath(WIKI_DOC, phase, 'simple')

#     save_complex_path = get_data_filepath(WIKI_DOC_FILTER, phase, 'complex')
#     save_simple_path = get_data_filepath(WIKI_DOC_FILTER, phase, 'simple')
#     #tmp = []
#     for complex_sent, simple_sent in (yield_sentence_pair(complex_file, simple_file)):
#         #tmp.append(len(tokenize(complex_sent))/len(tokenize(simple_sent)))
#         a = len(tokenize(complex_sent))
#         b = len(tokenize(simple_sent))
#         if b>256 or a>3000:
#             continue
#         #print(f'Complex lens: {a}, simple lens: {b}')
#         complex_lens.append(complex_sent)
#         simple_lens.append(simple_sent)
    
#     file_write_obj = open(save_complex_path, 'w', encoding='utf-8')
#     for var in complex_lens:
#         file_write_obj.write(var)
#         file_write_obj.write('\n')
#     file_write_obj.close()

#     file_write_obj = open(save_simple_path, 'w', encoding='utf-8')
#     for var in simple_lens:
#         file_write_obj.write(var)
#         file_write_obj.write('\n')
#     file_write_obj.close()

#     print("done")




# complex_lens = np.array(complex_lens)
# simple_lens = np.array(simple_lens)

# print(f'complex mean: {np.mean(complex_lens)}, simple mean: {np.mean(simple_lens)}')
# print(f'complex std: {np.std(complex_lens)}, simple std: {np.std(simple_lens)}')
# print(f'Len: {len(simple_lens)}')

# plt.hist(complex_lens, bins = 30)
# plt.hist(simple_lens, bins = 30)
# plt.title('distribution')
# plt.savefig('dis.png')





