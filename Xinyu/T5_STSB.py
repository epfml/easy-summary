from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizerFast, T5Tokenizer
import torch
device = 'cuda'
summarizer = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')


sent = ['Our story is amazing and a lot of people love it.']
tgt = ['The story is good and people like it.']

inputs = tokenizer(
    sent,
    max_length = 256,
    truncation = True,
    padding = 'max_length',
    return_tensors = 'pt'
).to(device)

src_ids = inputs['input_ids'].to(device)
src_mask = inputs['attention_mask'].to(device)


mid_ids = summarizer.generate(
    src_ids,
    num_beams=1, min_length = 3,
    max_length=20,
).to(device)

print(mid_ids)
print(mid_ids.shape)

attention_mask = torch.ones(mid_ids.shape).to(device)
print(tokenizer.pad_token_id)
attention_mask[mid_ids[:,:]==tokenizer.pad_token_id]=0

tgt = tokenizer(
    tgt,
    max_length = 256,
    truncation = True,
    padding = 'max_length',
    return_tensors = 'pt'
).to(device)

labels = tgt['input_ids'].to(device)
decoder_attention_mask = tgt['attention_mask'].to(device)

tmpids = summarizer.generate(
    mid_ids,
    num_beams=1, min_length = 3,
    max_length=20,
).to(device)

print(tokenizer.batch_decode(tmpids, skip_special_tokens = True, clean_up_tokenization_spaces = True))

outputs = summarizer(
    input_ids = mid_ids,
    attention_mask = attention_mask,
    labels = labels,
    decoder_attention_mask = decoder_attention_mask
)

print(outputs.loss)


