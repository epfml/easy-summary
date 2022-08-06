from preprocessor import DATASETS_DIR,PROCESSED_DATA_DIR,WIKI_DOC,WIKI_PARAGH_FILTER_DATASET, WIKILARGE_DATASET,yield_sentence_pair,yield_lines, WIKI_PARA_DATASET, get_data_filepath, tokenize, write_lines
import numpy as np
import tqdm
from summarizer import Summarizer
from preprocessor import WIKILARGE_DATASET
from rouge import Rouge

model = Summarizer(model='distilbert-base-uncased')
TYPES = ['complex', 'simple']
PHASES = ['train','valid','test']
HASH = '26ebb6aa762eac859c7b417fbb503eb7'

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
                

all_list = []

for phase in tqdm.tqdm(PHASES):
    if phase == 'test':
        continue
    complex_file = get_data_filepath(WIKI_DOC, phase, 'complex')
    simple_file = get_data_filepath(WIKI_DOC, phase, 'simple')
    tmp = []
    for complex_sent, simple_sent in (yield_sentence_pair(complex_file, simple_file)):
        tmp.append(len(tokenize(complex_sent))/len(tokenize(simple_sent)))
    all_list.append(np.array(tmp).mean())
print(all_list)
# [31.73652572747933, 31.64796329047812]

for phase in tqdm.tqdm(PHASES):
    '''
    Apply BERT to the complex sentences and get the summary.
    Ratio: 0.07
    '''
    if phase == 'test':
        continue
    complex_file_path = get_data_filepath(WIKI_DOC, phase, 'complex')
    save_complex_summary_path = get_data_filepath(WIKI_DOC, phase, 'complex_summary')
    summary = []
    for complex_sent in yield_lines(complex_file_path):
        complex_summary = model(complex_sent, ratio=0.07)
        summary.append(complex_summary)
    
    file_write_obj = open(save_complex_summary_path, 'w', encoding='utf-8')
    for var in tqdm.tqdm(summary):
        file_write_obj.write(var)
        file_write_obj.write('\n')
    file_write_obj.close()

print("Done!")


