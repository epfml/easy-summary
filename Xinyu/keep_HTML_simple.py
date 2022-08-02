from summarizer import Summarizer
from transformers import BartTokenizer, pipeline
from transformers import BartForConditionalGeneration
import pandas as pd
import torch
from preprocessor import Preprocessor
from Ts_T5 import T5FineTuner
# import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from bs4 import BeautifulSoup

def generate_simple_doc(doc):
    preprocessor = Preprocessor(features_kwargs)
    split_list = sent_tokenize(doc)
    gen_list = []
    for seq in split_list:
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

res = {"document": [], "simple_bert_0.6": []}
# Load our model checkpoints
print("######### Load models #########")
### BRIO Model
# model = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased').to(device)
# tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')

### BART
# BART_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

### BERT
BERT_summarizer = Summarizer(model='distilbert-base-uncased')
ratio = 0.6

############# generate simplified summary #############
# model_dirname = 'exp_paper'#'exp_turk_2loss_20_0.3prob_MeanComplexity'
# checkpoint_path = 'checkpoint-epoch=4.ckpt'

Simplified_Model = T5FineTuner.load_from_checkpoint('/mlodata1/blinova/text_sum_sim/easy-summary/Xinyu/experiments/exp_paper/checkpoint-epoch=4.ckpt')
simplified_model = Simplified_Model.model.to(device)
simplified_model_tokenizer = Simplified_Model.tokenizer#.to(device)

features_kwargs = {
    # 'WordRatioFeature': {'target_ratio': 1.05},
    'CharRatioFeature': {'target_ratio': 0.96},
    'LevenshteinRatioFeature': {'target_ratio': 0.78},
    'WordRankRatioFeature': {'target_ratio': 0.93},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.74}
}

df = pd.read_csv('resources/datasets/epfl_news_en/epfl_news_en.csv')
for num in tqdm(range(df.shape[0])):
    # soup_brio = BeautifulSoup(df.document.iloc[num], 'html.parser')
    # soup_bart = BeautifulSoup(df.document.iloc[num], 'html.parser')
    soup_bert = BeautifulSoup(df.document.iloc[num], 'html.parser')
    # soup_document = BeautifulSoup(df.document.iloc[num], 'html.parser')
    
    titles = []
    for title in soup_bert.find_all('strong'):
        titles.append(title.string)

    pars = []
    for par in soup_bert.p:
        if par.string is not None:
            pars.append(par.string)

    for title in titles:
        if title in pars:
            pars.remove(title)
    #--------------------------

    for article in tqdm(pars):
        # max_length = 512
        # BRIO
        # inputs = tokenizer([article], max_length=max_length, return_tensors="pt", truncation=True).to(device)
        # summary_ids = model.generate(inputs["input_ids"])
        # summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # summary_brio_simple = generate_simple_doc(summary)

        # for par in soup_brio.p:
        #     if par.string == article:
        #         par.string.replace_with(summary_brio_simple)
        #         break
        # BART
        # summary = BART_summarizer(article, min_length = 20, do_sample = False)[0]['summary_text']
        # summary_bart_simple = generate_simple_doc(summary)

        # for par in soup_bart.p:
        #     if par.string == article:
        #         par.string.replace_with(summary_bart_simple)
        #         break
        # BERT
        summary = BERT_summarizer(str(article), ratio = ratio)
        summary_bert_simple = generate_simple_doc(summary)

        for par in soup_bert.p:
            if par.string == article:
                par.string.replace_with(summary_bert_simple)
                break

        # # simplification only
        # document_simple = generate_simple_doc(article)

        # for par in soup_document.p:
        #     if par.string == article:
        #         par.string.replace_with(document_simple)
        #         break
        
        

    res["document"].append(df.document.iloc[num])
    # res["simple_document"].append(str(soup_document))
    # res["simple_brio"].append(str(soup_brio))
    # res['simple_bart'].append(str(soup_bart))
    res['simple_bert_0.6'].append(str(soup_bert))
    if num % 100 == 0:
        result = pd.DataFrame(res)
        result.to_csv('resources/datasets/epfl_news_en/simple_html.csv')

result = pd.DataFrame(res)
result.to_csv('resources/datasets/epfl_news_en/simple_html.csv')