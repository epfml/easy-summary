from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizerFast, T5Tokenizer
import torch
from easse.sari import corpus_sari
from transformers import T5ForConditionalGeneration
import torch.nn as nn

# import py7zr
# f1 = py7zr.SevenZipFile('Xinyu/resources/datasets/D_wiki/train.src.7z')
# f2 = py7zr.SevenZipFile('Xinyu/resources/datasets/D_wiki/train.tgt.7z')
# f1.extractall('Xinyu/resources/datasets/D_wiki/D_wiki.train.complex')
# f2.extractall('Xinyu/resources/datasets/D_wiki/D_wiki.train.simple')
# f1.close()
# f2.close()

device = 'cpu'
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')

# model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
# tokenizer = T5Tokenizer.from_pretrained('t5-base')

sent = ['simplify marouane fellaini and adnan januzaj continue to show \
the world they are not just teammates but also best mates. the manchester \
united and belgium duo both posted pictures of themselves out \
 at a restaurant on monday night ahead of their game against newcastle on wednesday . \
januzaj poses in the middle of fellaini and a friend looking like somebody who failed to receive the memo about it being a jackson 5 themed night.']
tg = ['The story is good and people like it.']



inputs = tokenizer(
    sent,
    max_length = 256,
    truncation = True,
    padding = 'max_length',
    return_tensors = 'pt'
).to(device)

src_ids = inputs['input_ids'].to(device)
src_mask = inputs['attention_mask'].to(device)

tgt = tokenizer(
    tg,
    max_length = 256,
    truncation = True,
    padding = 'max_length',
    return_tensors = 'pt'
).to(device)

labels = tgt['input_ids'].to(device)
labels[labels[:,:] == tokenizer.pad_token_id] = -100
decoder_attention_mask = tgt['attention_mask'].to(device)


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

# (1, 256, 768)
print(sum_outputs.encoder_last_hidden_state)
#print(sum_outputs.decoder_hidden_states[1].shape)

summary_ids = model.generate(
    src_ids,
    num_beams=10, min_length = 3,
    max_length=256,
).to(device)

# (1,98)
print(summary_ids)
print(summary_ids.shape)

padded_summary_ids = torch.zeros((summary_ids.shape[0], 256), dtype = torch.long).fill_(tokenizer.pad_token_id).to(device)

for i, summary_id in enumerate(summary_ids):
    print(summary_id.shape)
    padded_summary_ids[i, :summary_id.shape[0]] = summary_id

print(padded_summary_ids.shape)
attention_mask = torch.ones(padded_summary_ids.shape).to(device)
print(tokenizer.pad_token_id)
attention_mask[padded_summary_ids[:,:]==tokenizer.pad_token_id]=0


tmpids = model.generate(
    padded_summary_ids,
    num_beams=10, min_length = 3,
    max_length=256,
    do_sample=True,
                top_k=130,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=1
).to(device)
print(tmpids.shape)
# BART model
ans = tokenizer.batch_decode(tmpids, skip_special_tokens = True, clean_up_tokenization_spaces = True)[0]
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
# (1,98,768)
print(outputs.encoder_last_hidden_state)
#print(outputs.decoder_hidden_states[1].shape)


# W = torch.randn((768, int(768/2)), requires_grad=True).to(device)

# r1 = torch.matmul(sum_outputs.encoder_last_hidden_state, W)
# print(r1.shape)
# r2 = torch.matmul(outputs.encoder_last_hidden_state, W)
# print(r2.shape)

# sim = nn.CosineSimilarity(dim=2, eps=1e-6)
# score = sim(r1,r2)

# print(score.mean(dim = 1))

