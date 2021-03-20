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
        self.generator_config.hidden_size = config.generator_hidden_size
        self.generator_config.tokenizer = config.tokenizer
        self.generator_config.training_objective = (
            TrainingObjective.MaskedLanguageModelling
        )

        self.generator = VisualBERT(self.generator_config)

        self.discriminator_config = VisualBERTConfig()
        self.discriminator_config.bert_model = config.discriminator_model
        self.discriminator_config.hidden_size = config.discriminator_hidden_size
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
        generator_labels = generator_output.labels

        original_captions = batch[2]
        discriminator_output = self.discriminator.run_train_batch_discriminator(
            batch, generator_output
        )

        overall_loss = 0.5 * generator_output.loss + discriminator_output.loss

        accuracy = metrics.accuracy(
            discriminator_output.predictions, discriminator_output.targets
        )
        self.log(
            "generator_loss", overall_loss, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log(
            "discriminator_loss",
            overall_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        self.log(
            "train_loss", overall_loss, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log("epoch_train_loss", overall_loss, on_step=False, on_epoch=True)
        self.log(
            "train_discriminator_accuracy",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return overall_loss
