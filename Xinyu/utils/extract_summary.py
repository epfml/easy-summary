from summarizer import Summarizer
import nltk
from nltk.tokenize import sent_tokenize

import pandas as pd
from Ts_T5 import T5FineTuner
from sum_T5 import T5FineTuner_summary
import spacy
import re
nltk.download('punkt')
NLP = spacy.load('en_core_web_sm')

### convert the entity to tag
entity_to_tag = {}
tag_to_entity = {}
tag_to_url = {}
url_to_tag = {}
idx = 0
### replace the entity with the tag: ORG, PERSON

def Entity2Tag(Doc):
    '''
    replace the entity with the tag: ORG, PERSON
    '''
    doc = NLP(Doc)
    output = ""
    for tok in doc:
        text = tok.text
        flag = False
        if tok.ent_type_ in ['ORG', 'PERSON']:
            if tok.ent_type_ == 'ORG':
                if tok.text in entity_to_tag:
                    tag = entity_to_tag[tok.text]
                    output += tag + " "
                    flag = True
                else:
                    tag = 'ORG' + str(idx)
                    entity_to_tag[tok.text] = tag
                    tag_to_entity[tag] = tok.text
                    output += tag + ' '
                    flag = True
            else:
                if tok.text in entity_to_tag:
                    tag = entity_to_tag[tok.text]
                    output += tag + ' '
                    flag = True
                else:
                    tag = 'NAME' + str(idx)
                    entity_to_tag[tok.text] = tag
                    tag_to_entity[tag] = tok.text
                    output += tag + ' '
                    flag = True
        if tok.like_url:
            if tok.text in url_to_tag:
                tag = url_to_tag[tok.text]
                output += tag + ' '
                flag = True
            else:
                tag = 'URL' + str(idx)
                url_to_tag[tok.text] = tag
                tag_to_url[tag] = tok.text
                output += tag + ' '
                flag = True
        if flag:
            idx += 1
        else:
            output += text + ' '
    return output

def Tag2Entity(Doc):
    '''
    replace the tag with the entity
    '''
    doc  = NLP(Doc)
    output = ""
    for tok in doc:
        text = tok.text
        if text in tag_to_entity:
            output += tag_to_entity[text] + ' '
        elif text in tag_to_url:
            output += tag_to_url[text] + ' '
        else:
            output += text + ' '
    return output

def preprocess(df_path):
    df = pd.read_csv(df_path)
    df['preprocessed_doc'] = df['document'][:10].apply(lambda x: Entity2Tag(x))
    return df

df_path = 'Xinyu/resources/datasets/epfl_news/epfl_news_filtered_data_train.csv'
df = preprocess(df_path)
df.to_csv('Xinyu/resources/datasets/epfl_news/epfl_news_tagged.csv')



### Summarize tagged document on BERT
print("******** LOAD MODEL ********")
model = Summarizer(model='distilbert-base-uncased')
print('******** GENERATE SIMPLE DOCUMENT ********')

df = pd.read_csv('Xinyu/resources/datasets/epfl_news/epfl_news_tagged.csv')
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

def generate_summary(doc, ratio = 0.4):
    global cnt
    cnt+=1
    print(cnt)
    gen = model(doc,ratio=ratio)
    return gen

# cnt=0
# def generate_sum_T5(doc):
#     global cnt
#     cnt+=1
#     print(cnt)
#     gen = model2.generate(doc)
#     return gen

ratio = 0.4
df['BERT_summary_ratio_'+str(ratio)] = df.iloc[:10]['preprocessed_doc'].apply(lambda x:generate_summary(x,ratio))
df.to_csv('Xinyu/resources/datasets/epfl_news/epfl_news_tagged.csv')