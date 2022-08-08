from transformers import T5ForConditionalGeneration, T5TokenizerFast
import torch
from Ts_T5 import T5FineTuner

MODEL_NAME = 't5-small'
MODEL_PATH =  'Xinyu/experiments/exp_wikiparagh_10_epoch/checkpoint-epoch=3.ckpt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model = T5FineTuner.load_from_checkpoint(MODEL_PATH)
# model = Model.model.to(device)
# tokenizer = Model.tokenizer

tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

sent1 = ['The rhino grazed on the grass.']
sent2 = [' A rhino is grazing in a field.']

key_word = 'stsb'
inputs = []

for s1, s2 in zip(sent1, sent2):
    inputs.append(key_word+' sentence 1: '+ s1 + ' sentence 2: '+s2)

for ipt in inputs:
    input_ids = tokenizer(ipt, return_tensors = 'pt')['input_ids'].to(device)
    outputs = model.generate(input_ids)
    val = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print((val))

# from transformers import BertTokenizer, BertForPreTraining, BertModel
# import torch

# MODEL_NAME = 'bert-base-uncased'

# tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# #model = BertForPreTraining.from_pretrained(MODEL_NAME)
# model = BertModel.from_pretrained(MODEL_NAME)

# sent1 = 'Hello, my dog is cute.'
# sent2 = 'Hi, this is my dog and it is so cute.'

# sents = [sent1, sent2]

# input1 = tokenizer(sents, padding = True, return_tensors = 'pt')
# input2 = tokenizer(sent2, return_tensors = 'pt')
# output1 = model(**input1)

# h0 =  output1.last_hidden_state[0]
# h1 = output1.last_hidden_state[1]
# print(h0.shape, h1.shape)
# print(torch.bmm(h0, h1.transpose(1,2)).shape)



