from transformers import PreTrainedTokenizerBase, PreTrainedModel
from dataclasses import dataclass


@dataclass(init=False)
class VisualElectraConfig:
    generator_model: PreTrainedModel
    generator_hidden_size: int
    discriminator_model: PreTrainedModel
    discriminator_hidden_size: int
    tokenizer: PreTrainedTokenizerBase
    add_lm_head_discriminator: bool = False
