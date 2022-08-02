import spacy
import re
from spacy.tokenizer import Tokenizer
import pandas as pd

NLP = spacy.load('en_core_web_sm')

# complex_text = 'Convinced that the grounds were haunted http:/www/ , they decided to publish their findings in a book An Adventure ( 1911 ) , under the pseudonyms of Elizabeth Morison and Frances Lamont .'
# text1 = NER(complex_text)

# for word in text1.ents:
#     print(word.text, word.label_)
entity_to_tag = {}
tag_to_entity = {}
tag_to_url = {}
url_to_tag = {}

### replace the entity with the tag: ORG, PERSON

def Entity2Tag(Doc):
    '''
    replace the entity with the tag: ORG, PERSON
    '''
    doc = NLP(Doc)
    cnt = 0
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
                    tag = 'ORG' + str(cnt)
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
                    tag = 'NAME' + str(cnt)
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
                tag = 'URL' + str(cnt)
                url_to_tag[tok.text] = tag
                tag_to_url[tag] = tok.text
                output += tag + ' '
                flag = True
        if flag:
            cnt += 1
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

