from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizerFast, T5Tokenizer
import torch
from easse.sari import corpus_sari
from transformers import T5ForConditionalGeneration
device = 'cuda'
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')

# model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
# tokenizer = T5Tokenizer.from_pretrained('t5-base')

sent = ['Our story is amazing and a lot of people love it.']
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


mid_ids = model.generate(
    src_ids,
    num_beams=10, min_length = 3,
    max_length=20,
).to(device)

print(mid_ids)
print(mid_ids.shape)

attention_mask = torch.ones(mid_ids.shape).to(device)
print(tokenizer.pad_token_id)
attention_mask[mid_ids[:,:]==tokenizer.pad_token_id]=0

tgt = tokenizer(
    tg,
    max_length = 256,
    truncation = True,
    padding = 'max_length',
    return_tensors = 'pt'
).to(device)

labels = tgt['input_ids'].to(device)
decoder_attention_mask = tgt['attention_mask'].to(device)

tmpids = model.generate(
    mid_ids,
    num_beams=10, min_length = 3,
    max_length=20,
    do_sample=True,
                top_k=130,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=1
).to(device)
print(tmpids.shape)
ans = tokenizer.batch_decode(tmpids, skip_special_tokens = True, clean_up_tokenization_spaces = True)[0]
print(ans)

outputs = model(
    input_ids = mid_ids,
    attention_mask = attention_mask,
    labels = labels,
    decoder_attention_mask = decoder_attention_mask
)

print(outputs.loss)
print(corpus_sari(sent, [ans], [tg]))

