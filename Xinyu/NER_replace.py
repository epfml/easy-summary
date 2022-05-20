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
url_to_tag = {}
tag_to_url = {}
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
            entity_to_tag[tok.text] = tok.ent_type_ + str(cnt)
            tag_to_entity[tok.ent_type_ + str(cnt)] = tok.text
            output += tok.ent_type_ + str(cnt) + tok.whitespace_
            flag = True
        if tok.like_url:
            url_to_tag[tok.text] = 'URL' + str(cnt)
            tag_to_url['URL' + str(cnt)] = tok.text
            output += 'URL' + str(cnt) + tok.whitespace_
            flag = True
        if flag:
            cnt += 1
        else:
            output += text + tok.whitespace_
    return output

df = pd.read_csv('Xinyu/resources/datasets/epfl_news/epfl_news_filtered_data_train.csv')
for idx, row in df.iterrows():
    text = row['document']
    text = Entity2Tag(text)
    print(text)
    break
    

