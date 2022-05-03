import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import pandas as pd
from Ts_T5 import T5FineTuner
from sum_T5 import T5FineTuner_summary

print("******** LOAD MODEL ********")
model = T5FineTuner.load_from_checkpoint('experiments/exp_wikilarge_no_tokens_10epoch/checkpoint-epoch=8.ckpt')
print('******** GENERATE SIMPLE DOCUMENT ********')
df = pd.read_csv("resources/datasets/epfl_news/output_summary_BERT_T5.csv")
#df = pd.read_csv('resources/datasets/epfl_news/epfl_news_filtered_data_train.csv')
cnt = 0

def generate_simple_document(doc):
    global cnt
    split_lst = sent_tokenize(doc)
    gen_list = []
    cnt+=1
    print(cnt)
    
    for i,seq in enumerate(split_lst):
        print(i)
        gen = model.generate(seq)
        gen_list.append(gen)
    return " ".join(gen_list)



df['sent2sent_no_special_tokens_10epoch'] = df.iloc[:10]['BERT_summary_ratio_0.6'].apply(lambda x:generate_simple_document(x))
df.to_csv('resources/datasets/epfl_news/output_summary_BERT_T5.csv')