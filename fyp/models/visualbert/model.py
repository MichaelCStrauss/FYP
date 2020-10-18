import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from transformers import AutoModelForMaskedLM, AutoTokenizer
import pytorch_lightning.metrics.functional as metrics
from .config import VisualBERTConfig, TrainingObjective


class VisionLanguageEmbeddings(nn.Module):
    def __init__(self, hidden_dimensions, bert_embeddings: nn.Module, device="cuda"):
        super().__init__()
        self.device = device
        self.word_embeddings = bert_embeddings
        self.feature_dimensions = 1024
        self.hidden_dimensions = 512
        self.visual_projection = nn.Linear(
            self.feature_dimensions, self.hidden_dimensions
        )
        # 0 indicates text, 1 indicates vision
        self.token_type_embeddings = nn.Embedding(
            2, self.hidden_dimensions, padding_idx=0
        )
        self.embedding_layer_norm = nn.LayerNorm(self.hidden_dimensions)
        self.embedding_dropout = nn.Dropout(0.3)

    def forward(
        self, input_ids, attention_mask, vision_features, vision_mask, mask_indices=None
    ):
        # Let N be the padded sequence length e.g. if the longest string pre-tokenizing
        # was 15 tokens long, N=17 (one start token, one end token)
        if mask_indices is None:
            mask_indices = torch.zeros_like(input_ids) == 1

        # Don't mask the original input_ids as it is used for labels later
        # [8, N]
        masked_inputs = input_ids.clone().detach().to(self.device)
        # Mask token is 103 in BERT
        masked_inputs[mask_indices] = 103

        # Get the embeddings for each token id
        # [8, N, 512]
        text_embedding = self.word_embeddings(masked_inputs)
        # Prepare the inputs for the token type embedding
        # 0 is the padding index, lets define 1 as the 'text' type
        # Use a [8, N] matrix of ones with the padding tokens set to 0
        type_embedding_input = torch.ones_like(input_ids).long().to(self.device)
        type_embedding_input[attention_mask == 0] = 0

        # Get the embeddings, [8, N, 512]
        # token_type_embeddings = self.token_type_embeddings(type_embedding_input)

        # Sum the two types of embeddings
        # text_embedding = text_embedding + token_type_embeddings

        #
        # Vision
        #

        # `features` is a [8, 5, 1024] FloatTensor
        # Use a linear projection to convert to [8, 5, 512] to use as embeddings
        # Not sure if I should use an activation function here?
        features_embedding = self.visual_projection(vision_features)

        # Exact same process as before, except use 2 instead of 1 to indicate these
        # tokens are vision tokens
        type_embedding_input = (
            torch.zeros(features_embedding.shape[0:2]).long().to(self.device)
        )
        type_embedding_input[vision_mask == 0] = 0
        type_embedding_input[vision_mask == 1] = 2

        # Embeddding is still [8, 5, 512]
        # features_embedding = features_embedding + self.token_type_embeddings(
        #     type_embedding_input
        # )

        # Cat the two sequences: [8, N+5, 512]
        embeddings = torch.cat([features_embedding, text_embedding], dim=1)

        # Apply layer norm and droupout
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        # Make one attention mask as the cat of:
        # Text attention mask, vision attention mask, and then the [SEP] element
        # [8, N+6]
        attention_mask = torch.cat(
            [
                vision_mask,
                attention_mask,
            ],
            dim=1,
        )

        return embeddings, attention_mask


class VisualBERT(pl.LightningModule):
    def __init__(self, config: VisualBERTConfig, learning_rate: float = None):
        super().__init__()

        # Utils
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.uniform = torch.distributions.Uniform(0, 1)

        self.training_objective = config.training_objective
        self.tokenizer = config.tokenizer
        self.bert = config.bert_model

        self.hidden_dimensions = config.hidden_size
        self.embeddings = VisionLanguageEmbeddings(
            self.hidden_dimensions, self.bert.get_input_embeddings()
        )

    def setup(self, stage):
        pass

    def split_captions(self, features, vision_mask, caption_sets):
        num_captions_per_sample = len(caption_sets)
        batch_size = features.shape[0]
        features = features.repeat(num_captions_per_sample, 1, 1)
        vision_mask = vision_mask.repeat(num_captions_per_sample, 1)
        caption_sets = [
            c[i] for c in caption_sets for i in range(batch_size)
        ]
        return features, vision_mask, caption_sets

    def prepare_inputs_mlm(self, captions, features, vision_mask):
        # Transform a batch of captions and a batch of features into
        # model inputs, including mask

        #
        # Text
        #

        # Convert a 8-tuple of strings to an 8-element list of strings
        captions = list(captions)

        # Use the pre-trained weights from BERT for the text to begin with
        # These are adjusted by gradient descent

        # Tokenize the strings and include padding
        batch_encoding = self.tokenizer(
            captions,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_attention_mask=True,
        )

        # [8, N] LongTensor (token ids e.g. 'dog' might be 1000)
        input_ids = batch_encoding.input_ids.to(self.device)
        # [8, N] FloatTensor, mask for attention use later. 1 for tokens to attend,
        # 0 for padding tokens
        attention_mask = batch_encoding.attention_mask.to(self.device)

        # Text masking
        # Produce an [8, N] mask (True indicating tokens to mask)
        mask = self.mask(input_ids, batch_encoding.special_tokens_mask)
        # When BERT calculates loss with the labels, it does not compute loss for
        # elements with ID < 0. They use -100 as an example in the docs.
        # This ensures that all the un-masked outputs are not used
        # [8, N]
        text_labels = input_ids.clone().detach().to(self.device)
        text_labels[~mask] = -100

        # Useful for later
        text_length = input_ids.shape[1]

        embeddings, attention_mask = self.embeddings(
            input_ids, attention_mask, features, vision_mask, mask
        )

        # Prep the labels for loss
        # Make sure the shape is right: [8, N+6]
        # otherwise BERT gets angry
        labels = torch.zeros(embeddings.shape[0:2]).to(self.device)
        # The last N tokens are the same as previously computed
        labels[:, -text_length:] = text_labels
        # No prediction to be made on the vision tokens, so let it be -100
        labels[:, :-text_length] = -100
        labels = labels.long()

        return embeddings, attention_mask, labels

    def shift_labels(self, labels):
        """Shift labels, which include vision labels as -100, to be
        targets for CLM."""
        skip = (labels[0, :] == -100).sum().item()
        shifted_labels = labels[:, skip + 1 :].contiguous()
        return shifted_labels

    def prepare_inputs_clm(self, captions, features, vision_mask):
        # Transform a batch of captions and a batch of features into
        # model inputs, including mask

        #
        # Text
        #

        # Convert a 8-tuple of strings to an 8-element list of strings
        captions = list(captions)

        # Use the pre-trained weights from BERT for the text to begin with
        # These are adjusted by gradient descent

        # Tokenize the strings and include padding
        batch_encoding = self.tokenizer(
            captions,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_attention_mask=True,
        )

        # [8, N] LongTensor (token ids e.g. 'dog' might be 1000)
        input_ids = batch_encoding.input_ids.to(self.device)
        # [8, N] FloatTensor, mask for attention use later. 1 for tokens to attend,
        # 0 for padding tokens
        attention_mask = batch_encoding.attention_mask.to(self.device)

        embeddings, attention_mask = self.embeddings(
            input_ids, attention_mask, features, vision_mask
        )

        # Prep the labels for loss
        # Make sure the shape is right: [8, N+6]
        # otherwise BERT gets angry
        text_length = input_ids.shape[1]
        labels = torch.zeros(embeddings.shape[0:2]).to(self.device)
        # The last N tokens are the same as previously computed
        labels[:, -text_length:] = input_ids
        # No prediction to be made on the vision tokens, so let it be -100
        labels[:, :-text_length] = -100
        labels = labels.long()

        return embeddings, attention_mask, labels

    def forward(self, embeddings, attention_mask, labels=None):
        if self.training_objective == TrainingObjective.MaskedLanguageModelling:
            output = self.bert(
                inputs_embeds=embeddings,
                labels=labels,
                return_dict=True,
                attention_mask=attention_mask,
            )

            logits = output.logits
            loss = output.loss

            return logits, loss

        elif self.training_objective == TrainingObjective.Captioning:
            output = self.bert(
                inputs_embeds=embeddings,
                return_dict=True,
                attention_mask=attention_mask,
            )

            logits = output.logits
            loss = None
            if labels is not None:
                # we are doing next-token prediction;
                # shift prediction scores and input ids by one
                skip = (labels[0, :] == -100).sum().item()
                shifted_logits = logits[:, skip:-1, :].contiguous()
                shifted_labels = self.shift_labels(labels)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shifted_logits.view(-1, self.bert.config.vocab_size),
                    shifted_labels.view(-1),
                )

            return logits, loss

    def inference(self, vision_features, vision_mask, max_length):
        """ Greedy search using a language model approach to generate examples. """
        caption = "[CLS]"

        batch_encoding = self.tokenizer(
            [caption],
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_attention_mask=True,
            add_special_tokens=False,
        )
        # [1, N] LongTensor (token ids e.g. 'dog' might be 1000)
        input_ids = batch_encoding.input_ids.to(self.device)
        # [1, N] FloatTensor, mask for attention use later. 1 for tokens to attend
        # , 0 for padding tokens
        text_attention_mask = batch_encoding.attention_mask.to(self.device)

        while input_ids.shape[1] < max_length and input_ids[0, -1].item() != 102:
            embeddings, attention_mask = self.embeddings(
                input_ids, text_attention_mask, vision_features, vision_mask
            )

            logits, _ = self.forward(embeddings, attention_mask)

            next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            text_attention_mask = torch.cat(
                [text_attention_mask, torch.ones((1, 1)).to(self.device)], dim=1
            )

        decoded = self.tokenizer.decode(input_ids.squeeze())

        return decoded

    def mask(self, input_tokens, special_tokens, random=True):
        if random:
            mask = self.uniform.sample(input_tokens.shape) < 0.15
            mask = mask & (special_tokens == 0)
            return mask
        else:
            mask = torch.zeros_like(input_tokens)
            mask[:, 3] = 1
            mask[:, 6] = 1
            return mask == 1

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=(self.learning_rate if self.learning_rate is not None else 5e-5),
        )

    def training_step(self, batch, batch_idx):
        features, mask, caption_sets = batch
        features, vision_mask, caption = self.split_captions(
            features, mask, caption_sets
        )

        if self.training_objective == TrainingObjective.MaskedLanguageModelling:
            input_embeddings, attention_mask, labels = self.prepare_inputs_mlm(
                caption, features, vision_mask
            )
            targets = labels[labels != -100]

            logits, loss = self(input_embeddings, attention_mask, labels)

            predictions = torch.argmax(logits[labels != -100, :], dim=1)

        elif self.training_objective == TrainingObjective.Captioning:
            input_embeddings, attention_mask, labels = self.prepare_inputs_clm(
                caption, features, vision_mask
            )
            targets = self.shift_labels(labels).view(-1)

            logits, loss = self(input_embeddings, attention_mask, labels)

            skip = (labels[0, :] == -100).sum().item()
            shifted_logits = logits[:, skip:-1, :].contiguous()

            predictions = torch.argmax(shifted_logits, dim=2).view(-1)

        accuracy = metrics.accuracy(predictions, targets)
        # print(f"{predictions=}, {targets=}")

        result = pl.TrainResult(minimize=loss)

        result.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        result.log("epoch_train_loss", loss, on_step=False, on_epoch=True)
        result.log(
            "train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True
        )

        return result

    def validation_step(self, batch, batch_idx):
        features, mask, caption_sets = batch
        features, vision_mask, caption = self.split_captions(
            features, mask, caption_sets
        )

        if self.training_objective == TrainingObjective.MaskedLanguageModelling:
            input_embeddings, attention_mask, labels = self.prepare_inputs_mlm(
                caption, features, vision_mask
            )
            targets = labels[labels != -100]

            logits, loss = self(input_embeddings, attention_mask, labels)

            predictions = torch.argmax(logits[labels != -100, :], dim=1)

        elif self.training_objective == TrainingObjective.Captioning:
            input_embeddings, attention_mask, labels = self.prepare_inputs_clm(
                caption, features, vision_mask
            )
            targets = self.shift_labels(labels).view(-1)

            logits, loss = self(input_embeddings, attention_mask, labels)

            skip = (labels[0, :] == -100).sum().item()
            shifted_logits = logits[:, skip:-1, :].contiguous()

            predictions = torch.argmax(shifted_logits, dim=2).view(-1)

        accuracy = metrics.accuracy(predictions, targets)
        table_data = [
            [
                self.tokenizer.decode(p.tolist()),
                self.tokenizer.decode(t.tolist()),
            ]
            for p, t in zip(
                predictions.split(1),
                targets.split(1),
            )
        ]

        accuracy = metrics.accuracy(predictions, targets)

        result = pl.EvalResult(checkpoint_on=loss)

        result.log("val_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        result.log("epoch_val_loss", loss, on_step=False, on_epoch=True)
        result.log("val_accuracy", accuracy, on_step=True, on_epoch=True)
        result.log(
            "examples",
            wandb.Table(columns=["Prediction", "Target"], data=table_data),
            on_step=True,
            on_epoch=False,
        )

        return result
