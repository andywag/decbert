from DecBert import BertDecModel
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch
from torch.optim import AdamW

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
config = AutoConfig.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
#config.num_hidden_layers = 2
#config.max_position_embeddings = 32
#config.encoder_depth = 3
#config.encoder_block_size = 32
#config.max_position_dimension = 32
#print("I'm alive and well", config)

batch_size = 512
device = "cuda:0"
from datasets import load_dataset, load_from_disk

#dataset = load_dataset('wikitext','wikitext-2-v1')['train']
dataset = load_from_disk('small.hf')
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], max_length=128, truncation=True, pad_to_max_length=True), batched=True)
columns = ['input_ids', 'attention_mask', 'target']
tokenized_dataset.set_format(type='torch', columns=columns)

model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size)
iter_loader = iter(loader)
for data in iter_loader:
    for k, v in data.items():
        data[k] = v.to(device)
    result = model(**data)
    optimizer.zero_grad()
    result[1].backward()
    optimizer.step()
    print(result[1])

print("A", result)

#columns = ['input_ids', 'attention_mask']
#tokenized_dataset.set_format(type='torch', columns=columns)

#print("A", tokenized_dataset)