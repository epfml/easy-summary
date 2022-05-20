from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16).cuda()

# the fast tokenizer currently does not work correctly
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", use_fast=False)


prompt = 'His next work , Saturday , follows an especially eventful day in the life of a successful neurosurgeon .'\
        + 'Or simply input:'

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

generated_ids = model.generate(input_ids)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
