from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch
from datasets import load_dataset
import torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


checkpoint = 'sentence-transformers/all-MiniLM-L6-v2'
device = "cuda:0"
batch_size = 256

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
config = AutoConfig.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

model.to(device)

dataset = load_dataset('wikitext','wikitext-2-v1')['train']
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], max_length=128, truncation=True, pad_to_max_length=True), batched=True)
columns = ['input_ids', 'attention_mask']
tokenized_dataset.set_format(type='torch', columns=columns)

loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size)
iter_loader = iter(loader)

embeddings_result = []
for data in iter_loader:
    with torch.no_grad():
        for k, v in data.items():
            data[k] = v.to(device)
        result = model(**data)
        embed = mean_pooling(result, data['attention_mask'])
        embed = F.normalize(embed, 2, 1)
        base = embed.cpu()
        for x in range(base.shape[0]):
            embeddings_result.append(base[x].tolist())
    print("H")

print("Done Decodimg")
tok = tokenized_dataset.add_column('target',embeddings_result)
tok.save_to_disk('small.hf')
print("A", len(loader))

