import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM, DistilBertModel


class VisualBERT(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Utils
        self.learning_rate = None
        self.uniform = torch.distributions.Uniform(0, 1)

        # Base BERT model
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )
        self.bert = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased").to(
            self.device
        )

        # Embeddings
        self.text_embedding = self.bert.get_input_embeddings()
        self.feature_dimensions = 1024
        self.hidden_dimensions = 768
        self.visual_projection = nn.Linear(
            self.feature_dimensions, self.hidden_dimensions
        )
        # 0 indicates text, 1 indicates vision
        self.token_type_embeddings = nn.Embedding(2, self.hidden_dimensions)
        self.embedding_layer_norm = nn.LayerNorm(self.hidden_dimensions)
        self.embedding_dropout = nn.Dropout(0.3)

    # def setup(self, stage):
    #     self.logger.experiment.add_tags(["captioning", "simple-rnn", "flickr8k"])
    def prepare_inputs(self, captions, features, vision_mask):
        # Transform a batch of captions and a batch of features into
        # model inputs, including mask

        #
        # Text
        #
        captions = list(captions)

        batch_encoding = self.tokenizer(
            captions,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_attention_mask=True,
        )
        input_ids = batch_encoding.input_ids.to(self.device)
        attention_mask = batch_encoding.attention_mask.to(self.device)

        # Text masking
        mask = self.mask(input_ids, batch_encoding.special_tokens_mask)
        masked = input_ids.clone().detach().to(self.device)
        masked[mask] = 103

        text_labels = input_ids.clone().detach().to(self.device)
        text_labels[~mask] = -100

        text_length = input_ids.shape[1]
        text_embedding = self.text_embedding(masked)
        text_embedding = text_embedding + self.token_type_embeddings(
            torch.zeros_like(input_ids).long().to(self.device)
        )

        #
        # Vision
        #
        features_embedding = self.visual_projection(features)
        features_embedding = features_embedding + self.token_type_embeddings(
            torch.ones(features_embedding.shape[0:2]).long().to(self.device)
        )

        input_embeddings = torch.cat([text_embedding, features_embedding], dim=1)
        # Add another [SEP] token at the end
        expanded_embeddings = torch.cat(
            [
                input_embeddings,
                input_embeddings[:, (text_length - 1) : text_length, :],
            ],
            dim=1,
        )

        labels = torch.zeros(expanded_embeddings.shape[0:2]).to(self.device)
        labels[:, 0:text_length] = text_labels
        labels[:, text_length:] = -100
        labels = labels.long()

        attention_mask = torch.cat(
            [
                attention_mask,
                vision_mask,
                torch.ones((attention_mask.shape[0], 1)).to(self.device),
            ],
            dim=1,
        )

        return expanded_embeddings, labels, attention_mask

    def forward(self, input_embeddings, attention_mask=None, labels=None):
        output = self.bert(
            inputs_embeds=input_embeddings,
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
        caption, features, vision_mask = batch

        input_embeddings, labels, attention_mask = self.prepare_inputs(
            caption, features, vision_mask
        )

        batch_targets = self.tokenizer.decode(labels[labels != -100].tolist())

        logits, loss = self(
            input_embeddings, labels=labels, attention_mask=attention_mask
        )

        batch_preds = self.tokenizer.decode(
            torch.argmax(logits[labels != -100, :], dim=1).tolist()
        )
        # predicted_index = torch.argmax(logits, dim=2)
        # first_sentence = masked[0, :].tolist()
        # predicted_first = predicted_index[0, :].tolist()
        # print(f'{self.tokenizer.decode(first_sentence)=} {self.tokenizer.decode(predicted_first)}=')

        return {
            "loss": loss,
            "log": {
                "train_loss": loss,
                "batch_targets": batch_targets,
                "batch_preds": batch_preds,
            },
        }

    # def validation_step(self, batch, batch_idx):
    #     features, captions, lengths, targets, _ = batch

    #     y_hat = self(features, captions, lengths)

    #     loss = F.cross_entropy(y_hat, targets)

    #     return {"loss": loss, "log": {"val_loss": loss}}
