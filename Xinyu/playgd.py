from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizerFast, T5Tokenizer
import torch
from easse.sari import corpus_sari
from transformers import T5ForConditionalGeneration,pipeline
import torch.nn as nn
from keybert import KeyBERT
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

comparer = SentenceTransformer('all-MiniLM-L6-v2')

kl_loss = nn.KLDivLoss(reduction = 'batchmean', log_target = True)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
device = 'cpu'
# tokenizer = AutoTokenizer.from_pretrained("ml6team/keyphrase-generation-t5-small-inspec")

# model = AutoModelForSeq2SeqLM.from_pretrained("ml6team/keyphrase-generation-t5-small-inspec").to(device)

# import py7zr
# f1 = py7zr.SevenZipFile('Xinyu/resources/datasets/D_wiki/train.src.7z')
# f2 = py7zr.SevenZipFile('Xinyu/resources/datasets/D_wiki/train.tgt.7z')
# f1.extractall('Xinyu/resources/datasets/D_wiki/D_wiki.train.complex')
# f2.extractall('Xinyu/resources/datasets/D_wiki/D_wiki.train.simple')
# f1.close()
# f2.close()


# model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
# tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')

model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# sent = """
# Keyphrase extraction is a technique in text analysis where you extract the
# important keyphrases from a document. Thanks to these keyphrases humans can
# understand the content of a text very quickly and easily without reading it
# completely. Keyphrase extraction was first done primarily by human annotators,
# who read the text in detail and then wrote down the most important keyphrases.
# The disadvantage is that if you work with a lot of documents, this process
# can take a lot of time. 

# Here is where Artificial Intelligence comes in. Currently, classical machine
# learning methods, that use statistical and linguistic features, are widely used
# for the extraction process. Now with deep learning, it is possible to capture
# the semantic meaning of a text even better than these classical methods.
# Classical methods look at the frequency, occurrence and order of words
# in the text, whereas these neural approaches can capture long-term
# semantic dependencies and context of words in a text.
# """.replace("\n", " ")
sent = ['summarize: The cat sits outside', 'summarize: The new movie is awesome']
tg = ['The dog plays in the garden', 'The new movie is so great']
kws = ['cat_0.5 sits_0.5 outside_0.5', 'movie_0.5 awesome_0.5 new_0.5']

#Compute embedding for both lists
# embeddings1 = comparer.encode(sent, convert_to_tensor=True)
# embeddings2 = comparer.encode(tg, convert_to_tensor=True)

# print(embeddings1.shape)
# #Compute cosine-similarities
# cosine_scores = util.cos_sim(embeddings1, embeddings2)
# print(cosine_scores)

# sim_tgt = 'jason thomas kenney pc mla ( born may 30 , 1968 ) is a canadian politician . he is the 18th premier of alberta since 30 april 2019 , and leader of the united conservative party in alberta since 2017.kenney was inspired to enter politics after having a short conversation with former prime minister john diefenbaker at an early age . '
# hf_model = pipeline("feature-extraction", model="distilbert-base-cased")

# model = KeyBERT()
# res = model.extract_keywords(sim_tgt, keyphrase_ngram_range=(1, 2), stop_words=None)
# print(res[0][0]+' '+ sim_tgt)

# inputs = tokenizer(
#     sent,
#     max_length = 256,
#     truncation = True,
#     padding = 'max_length',
#     return_tensors = 'pt'
# ).to(device)

# src_ids = inputs['input_ids'].to(device)
# src_mask = inputs['attention_mask'].to(device)

# o = model.generate(src_ids)

# tgt = tokenizer.decode(o[0], skip_special_tokens=True)

# decoding = tokenizer(
#     tgt,
#     max_length = 256,
#     truncation = True,
#     padding = 'max_length',
#     return_tensors = 'pt'
# )

# labels = decoding['input_ids'].to(device)
# labels[labels[:,:] == tokenizer.pad_token_id] = -100
# decoder_attention_mask = decoding['attention_mask'].to(device)

# print(tokenizer.decode(o[0], skip_special_tokens=True))

# outputs = model(
#     input_ids = src_ids,
#     attention_mask = src_mask,
#     labels = labels,
#     decoder_attention_mask = decoder_attention_mask,
#     #output_hidden_states = True
# )
# print(outputs.loss)

max_seq = 256

inputs = tokenizer(
    sent,
    max_length = max_seq,
    truncation = True,
    padding = 'max_length',
    return_tensors = 'pt'
).to(device)

src_ids = inputs['input_ids'].to(device)
src_mask = inputs['attention_mask'].to(device)

tgt = tokenizer(
    tg,
    max_length = max_seq,
    truncation = True,
    padding = 'max_length',
    return_tensors = 'pt'
).to(device)

labels = tgt['input_ids'].to(device)
labels[labels[:,:] == tokenizer.pad_token_id] = -100
decoder_attention_mask = tgt['attention_mask'].to(device)

kw_encoding = tokenizer(
    kws,
    max_length = max_seq,
    truncation = True,
    padding = 'max_length',
    return_tensors = 'pt'
).to(device)

kw_ids = kw_encoding['input_ids'].to(device)
print("kw_ids: ", kw_ids.shape)
print("kw_ids[0]: ", kw_ids[0])
print('decode: ', tokenizer.decode(kw_ids[0], skip_special_tokens=True))
# for src_id in src_ids:
#     # add tokens in front of the src_id
#     tokens = torch.tensor([18356, 10]).to(device)
#     src_id = torch.cat((tokens, src_id), dim=0)[:-2]
#     print(src_id.shape)

# print(src_ids)

sum_outputs = model(
    input_ids = src_ids,
    attention_mask = src_mask,
    labels = labels,
    decoder_attention_mask = decoder_attention_mask,
    output_hidden_states = True
)

# (1, 256, 768) --> (1,512, 768)
H1 = sum_outputs.encoder_last_hidden_state
print(sum_outputs.encoder_last_hidden_state.shape, H1.requires_grad)
#print(sum_outputs.decoder_hidden_states[1].shape)

summary_ids = model.generate(
    src_ids,
    num_beams=10, min_length = 3,
    max_length=256,
).to(device)

print('decode: ', tokenizer.batch_decode(summary_ids, skip_special_tokens=True))
 
# (1,98) --> (1,48)
# print(summary_ids)
# print(summary_ids.shape)

padded_summary_ids = torch.zeros((summary_ids.shape[0], max_seq), dtype = torch.long).fill_(tokenizer.pad_token_id).to(device)
for i, summary_id in enumerate(summary_ids):
    print(summary_id.shape)
    padded_summary_ids[i, :summary_id.shape[0]] = summary_id
    
for i, pad_summary_id in enumerate(padded_summary_ids):
    added = kw_ids[i]
    # get non-zero elements
    added = added[added != 0]
    #print(added)
    new_ = torch.cat((added,pad_summary_id), dim=0)[:max_seq]
    # add "simplify: " token
    padded_summary_ids[i] = torch.cat((torch.tensor([18356, 10]).to(device), new_), dim=0)[:max_seq]
   


print("padding summary_ids: ",padded_summary_ids.shape)
attention_mask = torch.ones(padded_summary_ids.shape).to(device)
print(tokenizer.pad_token_id)
attention_mask[padded_summary_ids[:,:]==tokenizer.pad_token_id]=0
print(tokenizer.batch_decode(padded_summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))


tmpids = model.generate(
    padded_summary_ids,
    num_beams=10, min_length = 3,
    max_length=max_seq,
    do_sample=True,
                top_k=130,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=1
).to(device)
print(tmpids.shape)
# T5 model
#ans = tokenizer.decode(tmpids[0], skip_special_tokens = True, clean_up_tokenization_spaces = True)
#print(ans)

outputs = model(
    input_ids = padded_summary_ids,
    attention_mask = attention_mask,
    labels = labels,
    decoder_attention_mask = decoder_attention_mask,
    output_hidden_states = True
)
# (1,512,768)
H2 = outputs.encoder_last_hidden_state
print(outputs.encoder_last_hidden_state.shape)
#print(outputs.decoder_hidden_states[1].shape)

# print(util.cos_sim(H2.mean(dim = 1),H1.mean(dim=1)))

# W = torch.randn((768, 1), requires_grad=True).to(device)
# Q = torch.randn((max_seq, 512), requires_grad = True).to(device)

# r1 = torch.transpose((torch.transpose(H1, 1,2) @ Q), 1,2)
# r2 = torch.transpose((torch.transpose(H2, 1,2) @ Q), 1,2)

# print(r1.shape, r2.shape)

# r1 = torch.matmul(r1, W)
# print(r1.shape)
# r2 = torch.matmul(r2, W)
# print(r2.shape)

# r1 = r1.squeeze(dim=2)
# opt = nn.LogSoftmax(dim=1) 
# print(opt(r1).shape)



# r1 = opt(r1)
# r2 = opt(r2)

# print(kl_loss(r1,r2))
# sim = nn.CosineSimilarity(dim=2, eps=1e-6)
# score = sim(r1,r2)

# print(score.mean(dim = 1))

