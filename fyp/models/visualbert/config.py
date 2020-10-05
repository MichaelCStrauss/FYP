from transformers import PreTrainedTokenizerBase, PreTrainedModel, BertModel
from dataclasses import dataclass
from enum import Enum


class TrainingObjective(Enum):
    MaskedLanguageModelling = "mlm"
    Captioning = "captioning"


@dataclass(init=False)
class VisualBERTConfig:
    bert_model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    hidden_size: int

    training_objective: TrainingObjective
