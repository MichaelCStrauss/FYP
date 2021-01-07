from transformers import PreTrainedTokenizerBase, PreTrainedModel
from dataclasses import dataclass


@dataclass(init=False)
class VisualElectraConfig:
    generator_model: PreTrainedModel
    discriminator_model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    hidden_size: int
