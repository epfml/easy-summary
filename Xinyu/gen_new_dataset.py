from preprocessor import DATASETS_DIR,PROCESSED_DATA_DIR,WIKI_DOC,WIKI_PARAGH_FILTER_DATASET, WIKILARGE_DATASET,yield_sentence_pair,yield_lines, WIKI_PARA_DATASET, get_data_filepath, tokenize, write_lines
import numpy as np
import tqdm
from summarizer import Summarizer

model = Summarizer(model='distilbert-base-uncased')
TYPES = ['complex', 'simple']
PHASES = ['train','valid','test']
HASH = '26ebb6aa762eac859c7b417fbb503eb7'

# all_list = []

# for phase in tqdm.tqdm(PHASES):
#     if phase == 'test':
#         continue
#     complex_file = get_data_filepath(WIKI_DOC, phase, 'complex')
#     simple_file = get_data_filepath(WIKI_DOC, phase, 'simple')
#     tmp = []
#     for complex_sent, simple_sent in (yield_sentence_pair(complex_file, simple_file)):
#         tmp.append(len(tokenize(complex_sent))/len(tokenize(simple_sent)))
#     all_list.append(np.array(tmp).mean())
# print(all_list)
#[31.73652572747933, 31.64796329047812]

for phase in tqdm.tqdm(PHASES):
    '''
    Apply BERT to the simple sentences and get the summary.
    Ratio: 0.07
    '''
    if phase == 'test':
        continue
    complex_file_path = get_data_filepath(WIKI_DOC, phase, 'complex')
    save_complex_summary_path = get_data_filepath(WIKI_DOC, phase, 'complex_summary')

    for complex_sent in yield_lines(complex_file_path):
        complex_summary = model(complex_sent, ratio=0.07)
        write_lines(complex_summary, save_complex_summary_path)

print("Done!")


