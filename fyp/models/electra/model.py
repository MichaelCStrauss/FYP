import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import pytorch_lightning.metrics.functional as metrics
from ..visualbert.config import VisualBERTConfig, TrainingObjective
from ..visualbert.model import VisualBERT
from .config import VisualElectraConfig


class VisualElectra(pl.LightningModule):
    def __init__(self, config: VisualElectraConfig, learning_rate: float = None):
        super().__init__()
        # Utils
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.generator_config = VisualBERTConfig()
        self.generator_config.bert_model = config.generator_model
        self.generator_config.hidden_size = config.hidden_size
        self.generator_config.tokenizer = config.tokenizer
        self.generator_config.training_objective = (
            TrainingObjective.MaskedLanguageModelling
        )

        self.generator = VisualBERT(self.generator_config)

        self.discriminator_config = VisualBERTConfig()
        self.discriminator_config.bert_model = config.discriminator_model
        self.discriminator_config.hidden_size = config.hidden_size
        self.discriminator_config.tokenizer = config.tokenizer
        self.discriminator_config.training_objective = TrainingObjective.Discriminator

        self.discriminator = VisualBERT(self.discriminator_config)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=(self.learning_rate if self.learning_rate is not None else 5e-5),
        )

    def training_step(self, batch, batch_idx):
        generator_output = self.generator.run_train_batch(batch)

        # Prepare labels
        return {"loss": generator_output.loss}
