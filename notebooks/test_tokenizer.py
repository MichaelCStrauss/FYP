# %%
import torch
from fyp.data.oscar_data import CaptionTSVDataset
from fyp.models.oscar.pytorch_transformers.tokenization_bert import BertTokenizer
from tqdm import tqdm

# %%
original_tokenizer = BertTokenizer.from_pretrained(
    "/home/michael/Documents/FYP/models/oscar/coco_captioning_base_scst/checkpoint-15-66405"
)

small_tokenizer = BertTokenizer.from_pretrained(
    "/home/michael/Documents/FYP/models/bert_tiny_vocab"
)

# %%

test_sentence = "A black Honda motorcycle parked in front of a garage."

print(original_tokenizer.tokenize(test_sentence))
print(small_tokenizer.tokenize(test_sentence))

# %%
original_dataset = CaptionTSVDataset(
    '/home/michael/Documents/FYP/data/processed/coco-oscar/test.yaml',
    tokenizer=original_tokenizer,
    add_od_labels=False,
    max_img_seq_length=50,
    max_seq_length=70,
    max_seq_a_length=40,
    is_train=True,
    mask_prob=0.15,
    max_masked_tokens=3,
)
small_dataset = CaptionTSVDataset(
    '/home/michael/Documents/FYP/data/processed/coco-oscar/test.yaml',
    tokenizer=small_tokenizer,
    add_od_labels=False,
    max_img_seq_length=50,
    max_seq_length=70,
    max_seq_a_length=40,
    is_train=True,
    mask_prob=0,
    max_masked_tokens=0,
)

# %%
(_, first_original), (_, first_small) = next(zip(iter(original_dataset), iter(small_dataset)))
o_input_ids = first_original[0]
s_input_ids = first_small[0]

masked_pos = first_original[4]
masked_ids = first_original[5]

expanded = torch.tensor(small_tokenizer.encode(original_tokenizer.decode(masked_ids.unsqueeze(0))))
print(expanded)
no_pad_ex = expanded[expanded != 0]
idx = (s_input_ids[..., None] == no_pad_ex).any(-1).nonzero()
s_input_ids[idx] = 103

print(original_tokenizer.decode(o_input_ids.tolist()))
print(small_tokenizer.decode(s_input_ids.tolist()))

# %%
same, different = 0, 0
for original, small in tqdm(zip(iter(original_dataset), iter(small_dataset))):
    id, example = original
    o_input_ids, *_ = example
    # original_sentence = original_tokenizer.decode(input_ids.tolist())
    id, example = small
    s_input_ids, *_ = example
    # small_sentence = small_tokenizer.decode(input_ids.tolist())

    if torch.all(torch.eq(o_input_ids, s_input_ids)):
        same += 1
    else:
        different += 1

print(same)
print(different)
# %%

# %%
