import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import pandas as pd
from Ts_T5 import T5FineTuner
from sum_T5 import T5FineTuner_summary

print("******** LOAD MODEL ********")
model = T5FineTuner.load_from_checkpoint('Xinyu/experiments/exp_turk_3loss_3_10/checkpoint-epoch=1.ckpt')
print('******** GENERATE SIMPLE DOCUMENT ********')
df = pd.read_csv("Xinyu/resources/datasets/epfl_news/output_summary_BERT_0.4.csv")
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



df['sent2sent_newloss_turk_20epoch_3_10'] = df.iloc[:10]['BERT_summary_ratio_0.4'].apply(lambda x:generate_simple_document(x))
df.to_csv('Xinyu/resources/datasets/epfl_news/output_summary_BERT_0.4.csv')