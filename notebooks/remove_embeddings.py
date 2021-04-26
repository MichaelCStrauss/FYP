# %%
import torch

ckpt = torch.load('/home/michael/Documents/FYP/models/bert_tiny_vocab2/pytorch_model.bin')

for key in ['bert.embeddings.word_embeddings.weight', 'cls.predictions.bias', 'cls.predictions.decoder.weight']:
    del ckpt[key]

torch.save(ckpt, '/home/michael/Documents/FYP/models/bert_tiny_vocab2/pytorch_model.bin')

# %%
