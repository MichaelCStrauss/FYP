# %%
import torch
from transformers import DistilBertTokenizerFast

# %%
u = torch.distributions.Uniform(0, 1)
# %%
test = torch.range(10, 29).reshape((2, -1))
r = u.sample(test.shape)
masked = r < 0.15

test[masked] = -100
test

# %%
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokenizer('A [MASK] is walking down the street', pad_to_multiple_of=12, padding=True)

# %%
