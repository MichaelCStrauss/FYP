import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM, DistilBertModel


class VisualBERT(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.learning_rate = None
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )
        self.bert = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased").to(
            self.device
        )
        self.uniform = torch.distributions.Uniform(0, 1)

    # def setup(self, stage):
    #     self.logger.experiment.add_tags(["captioning", "simple-rnn", "flickr8k"])

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.bert(
            input_ids=input_ids,
            labels=labels,
            return_dict=True,
            attention_mask=attention_mask,
        )

        logits = output.logits
        loss = output.loss

        return logits, loss

    def mask(self, input_tokens, special_tokens):
        mask = self.uniform.sample(input_tokens.shape) < 0.15
        mask = mask & (special_tokens == 0)
        return mask

    def configure_optimizers(self):
        return transformers.AdamW(
            self.parameters(),
            lr=(self.learning_rate if self.learning_rate is not None else 1e-5),
        )

    def training_step(self, batch, batch_idx):
        caption = batch

        batch_encoding = self.tokenizer(
            caption, padding=True, return_tensors="pt", return_special_tokens_mask=True
        )
        input_ids = batch_encoding.input_ids
        attention_mask = batch_encoding.input_ids.to(self.device)
        mask = self.mask(input_ids, batch_encoding.special_tokens_mask)

        masked = input_ids.clone().detach().to(self.device)
        masked[mask] = 103

        labels = input_ids.clone().detach().to(self.device)
        labels[~mask] = -100

        logits, loss = self(masked, labels=labels, attention_mask=attention_mask)
        # predicted_index = torch.argmax(logits, dim=2)
        # first_sentence = masked[0, :].tolist()
        # predicted_first = predicted_index[0, :].tolist()
        # print(f'{self.tokenizer.decode(first_sentence)=} {self.tokenizer.decode(predicted_first)}=')

        return {"loss": loss, "log": {"train_loss": loss}}

    # def validation_step(self, batch, batch_idx):
    #     features, captions, lengths, targets, _ = batch

    #     y_hat = self(features, captions, lengths)

    #     loss = F.cross_entropy(y_hat, targets)

    #     return {"loss": loss, "log": {"val_loss": loss}}
