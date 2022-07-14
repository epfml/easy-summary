from pathlib import Path
import sys
# sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
import numpy as np
from summarizer import Summarizer
from transformers import BartTokenizer, PegasusTokenizer, pipeline
from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import torch
from preprocessor import EXP_DIR, Preprocessor
from Ts_T5 import T5FineTuner
# import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize

IS_CNNDM = True # whether to use CNNDM dataset or XSum dataset
LOWER = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

res = {"document": [], "summary_brio": [], "summary_bart": [], "summary_bert_0.55": []}
# Load our model checkpoints
### BRIO Model
if IS_CNNDM:
    model = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased').to(device)
    tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')
else:
    model = PegasusForConditionalGeneration.from_pretrained('Yale-LILY/brio-xsum-cased').to(device)
    tokenizer = PegasusTokenizer.from_pretrained('Yale-LILY/brio-xsum-cased')

### BART
BART_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
### T5
# T5_summarizer = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
# T5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

### BERT
BERT_summarizer = Summarizer(model='distilbert-base-uncased')
ratio = 0.55

df = pd.read_csv('Xinyu/resources/datasets/epfl_news/epfl_news_filtered_data_train.csv')
# number of sentences
n = 4
# number of examples in exp
for num in range(5):
    print('num: ', num)
    summary_brio = []
    summary_bart = []
    summary_bert = []
    doc = sent_tokenize(df.document.iloc[num])
    sents = [' '.join(doc[i:i+n]) for i in range(0, len(doc), n)]
    if len(doc) % n != 0:
        sents[-2:] = [' '.join(sents[-2:])]
    for ARTICLE in sents:
        max_length = 512
        # generation example
        if LOWER:
            article = ARTICLE.lower()
        else:
            article = ARTICLE
        inputs = tokenizer([article], max_length=max_length, return_tensors="pt", truncation=True).to(device)
        # Brio 
        summary_ids = model.generate(inputs["input_ids"])
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        summary_brio.append(summary)
        # BART
        summary_bart.append(BART_summarizer(article,min_length = 30, do_sample = False)[0]['summary_text'])
        # T5
        # t5_prepared_text = 'summarize: ' + ARTICLE
        # tokenized_text = T5_tokenizer.encode(t5_prepared_text, return_tensors="pt").to(device)
        # summary_ids = model.generate(
        #     tokenized_text,
        #     num_beams=10,
        #     no_repeat_ngram_size=2,
        #     min_length=30,
        #     max_length = 100,
        #     early_stopping=True,
        # )
        # output = T5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # summary_T5.append(output)

        # BERT
        summary_bert.append(BERT_summarizer(article, ratio = ratio))
        
    res["document"].append(' '.join(sents))
    res["summary_brio"].append(' '.join(summary_brio))
    res['summary_bart'].append(' '.join(summary_bart))
    res['summary_bert_0.55'].append(' '.join(summary_bert))
    # res['summary_t5_small'].append(' '.join(summary_T5))

result_cnndm = pd.DataFrame(res)
result_cnndm.to_csv('Xinyu/resources/datasets/epfl_news/summary_4sent.csv')

############# generate simplified summary #############
model_dirname = 'exp_turk_2loss_20_0.3prob_MeanComplexity'
checkpoint_path = 'checkpoint-epoch=1.ckpt'
df = pd.read_csv("Xinyu/resources/datasets/epfl_news/summary_4sent.csv")

print("######### Load model #########")
Simplified_Model = T5FineTuner.load_from_checkpoint(EXP_DIR / model_dirname / checkpoint_path).to(device)
simplified_model = Simplified_Model.model.to(device)
simplified_model_tokenizer = Simplified_Model.tokenizer

features_kwargs = {
    # 'WordRatioFeature': {'target_ratio': 1.05},
    'CharRatioFeature': {'target_ratio': 0.96},
    'LevenshteinRatioFeature': {'target_ratio': 0.78},
    'WordRankRatioFeature': {'target_ratio': 0.93},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.74}
}
cnt = 0
print("######### Generate simplified summary #########")

def generate_simple_doc(doc):
    preprocessor = Preprocessor(features_kwargs)
    split_list = sent_tokenize(doc)
    gen_list = []
    for i, seq in enumerate(split_list):
        print("{}/{}".format(i, len(split_list)))

        seq = preprocessor.encode_sentence(seq)
        text = 'simplify: ' + seq
        encoding = simplified_model_tokenizer(text,
                                              max_length = 256,
                                              padding = 'max_length',
                                              truncation = True,
                                              return_tensors = 'pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        beam_outputs = simplified_model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            do_sample = False,
            max_length = 256,
            num_beams = 10,
            top_k = 130,
            top_p = 0.97,
            early_stopping = True,
            num_return_sequences = 1,
        )
        gen_sent = simplified_model_tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        gen_list.append(gen_sent)
    
    return " ".join(gen_list)

df['simple_brio'] = df['summary_brio'].apply(lambda x:generate_simple_doc(x))
df['simple_bart'] = df['summary_bart'].apply(lambda x:generate_simple_doc(x))
df['simple_bert_0.55'] = df['summary_bert_0.55'].apply(lambda x:generate_simple_doc(x))
df['simple_document'] = df['document'].apply(lambda x:generate_simple_doc(x))
df.to_csv('Xinyu/resources/datasets/epfl_news/sum_simple_4sent.csv')


