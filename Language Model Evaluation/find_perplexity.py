import os
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

device = "cuda:1"

with open('') as f:
    final_sentences = f.readlines()

from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForMaskedLM.from_pretrained("roberta-base").to(device)

sentence_to_score = []
for s in tqdm(final_sentences[:10000]):
    if len(s.split(' ')) > 100 or len(s.split(' ')) < 5:
        continue
    model.eval()
    with torch.no_grad():
        tensor_input = tokenizer.encode(s.strip(), return_tensors='pt')
        if tensor_input.size(-1) > 128:
            continue
        repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
        labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
        masked_input = masked_input.to(device)
        labels = labels.to(device)
        loss = model(masked_input, labels=labels).loss.item()
        labels = labels.cpu().detach()
        masked_input = masked_input.cpu().detach()
        sentence_to_score.append((s, loss))

with open('', 'w') as f:
    for s, l in sentence_to_score:
        f.write(s+" || "+str(l)+'\n')