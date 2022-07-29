from SenBert import SenBert
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch
from torch.optim import AdamW

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
config = AutoConfig.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
true_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
true_dict = true_model.state_dict()


#print("I'm alive and well", config)

batch_size = 64
device = "cuda:0"
from datasets import load_dataset, load_from_disk

#dataset = load_dataset('wikitext','wikitext-2-v1')['train']
dataset = load_from_disk('small.hf')
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], max_length=128, truncation=True, pad_to_max_length=True), batched=True)
columns = ['input_ids', 'attention_mask', 'target']
tokenized_dataset.set_format(type='torch', columns=columns)

model = SenBert(config).to(device)
#model.bert.load_state_dict(true_dict)
params = model.parameters()
#for param in params:
#    print(param.shape)
optimizer = AdamW(params, lr=3e-6)

loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size)
iter_loader = iter(loader)

for x in range(10000):
    try:
        data = next(iter_loader)
    except:
        iter_loader = iter(loader)
        data = next(iter_loader)
#for data in iter_loader:
    for k, v in data.items():
        data[k] = v.to(device)
    result = model(**data)
    err = result[0][0] - data['target'][0]
    optimizer.zero_grad()
    result[1].backward()
    optimizer.step()
    print(result[1])

print("A", result)

#columns = ['input_ids', 'attention_mask']
#tokenized_dataset.set_format(type='torch', columns=columns)

#print("A", tokenized_dataset)