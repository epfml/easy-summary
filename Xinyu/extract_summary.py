from summarizer import Summarizer
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import pandas as pd
from Ts_T5 import T5FineTuner
from sum_T5 import T5FineTuner_summary

print("******** LOAD MODEL ********")
model = Summarizer(model='distilbert-base-uncased')
#model2 = T5FineTuner_summary.load_from_checkpoint('experiments/exp_epfl_summary_FineTune_2epoch/checkpoint-epoch=1.ckpt')
#model = T5FineTuner_summary.load_from_checkpoint("experiments/exp_epfl_summary_FineTune_10epoch/checkpoint-epoch=6.ckpt")
print('******** GENERATE SIMPLE DOCUMENT ********')

df = pd.read_csv('Xinyu/resources/datasets/epfl_news/epfl_news_filtered_data_train.csv')
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

def generate_summary(doc):
    global cnt
    cnt+=1
    print(cnt)
    gen = model(doc,ratio=0.4)
    #gen = model.generate(doc)
    return gen

# cnt=0
# def generate_sum_T5(doc):
#     global cnt
#     cnt+=1
#     print(cnt)
#     gen = model2.generate(doc)
#     return gen


df['BERT_summary_ratio_0.4'] = df.iloc[:10]['document'].apply(lambda x:generate_summary(x))
df.to_csv('Xinyu/resources/datasets/epfl_news/output_summary_BERT_0.4.csv')