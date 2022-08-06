# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
from preprocessor import yield_lines, get_data_filepath, yield_sentence_pair,tokenize,TURKCORPUS_DATASET, WIKI_DOC
from summarizer import Summarizer
import numpy as np


model = Summarizer(model='distilbert-base-uncased')
dataset = WIKI_DOC

complex_file_path = get_data_filepath(dataset,'test', 'complex')

### different ratios for BERT_Summarizer
p = [0.1,0.3,0.5,0.7,0.9]

for ratio in p:
    print("ratio:", ratio)
    save_complex_summary_path = get_data_filepath(dataset,'test', 'complex_summary_'+str(ratio))
    summary = []
    for complex_sent in yield_lines(complex_file_path):
        complex_summary = model(complex_sent, ratio=ratio)
        summary.append(complex_summary)

    file_write_obj = open(save_complex_summary_path, 'w', encoding='utf-8')
    for var in summary:
        file_write_obj.write(var)
        file_write_obj.write('\n')
    file_write_obj.close()
    print("done")

#### 
all_list = []
for ratio in p:
    complex_file = get_data_filepath(dataset,'test','complex')
    summary_file = get_data_filepath(dataset,'test','complex_summary_'+str(ratio))
    tmp =[]
    cnt = 0
    for complex_sent, summary_sent in yield_sentence_pair(complex_file, summary_file):
        if len(tokenize(summary_sent))==0:
            continue
        tmp.append(len(tokenize(complex_sent))/len(tokenize(summary_sent)))
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)
    all_list.append(np.array(tmp).mean())

# [6.215887984952227, 2.9000465574257315, 1.9466336457558744, 1.4818730730731375, 1.1727811268198567]