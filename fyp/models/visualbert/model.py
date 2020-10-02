import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer


class VisualBERT(pl.LightningModule):
    def __init__(self, learning_rate: float = None):
        super().__init__()

        # Utils
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.uniform = torch.distributions.Uniform(0, 1)

        # Base BERT model
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/bert_uncased_L-4_H-512_A-8"
            # "bert-base-uncased"
        )
        self.bert = AutoModelForMaskedLM.from_pretrained(
            "google/bert_uncased_L-4_H-512_A-8"
            # "bert-base-uncased"
        )
        self.bert.train()

        # Embeddings
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

    def setup(self, stage):
        pass

    def prepare_inputs(self, captions, features, vision_mask):
        # Transform a batch of captions and a batch of features into
        # model inputs, including mask

        #
        # Text
        #

        # Convert a 8-tuple of strings to an 8-element list of strings
        captions = list(captions)

        # Use the pre-trained weights from BERT for the text to begin with
        # These are adjusted by gradient descent
        text_embedding_layer = self.bert.get_input_embeddings()

        # Tokenize the strings and include padding
        batch_encoding = self.tokenizer(
            captions,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_attention_mask=True,
        )

        # Let N be the padded sequence length e.g. if the longest string pre-tokenizing
        # was 15 tokens long, N=17 (one start token, one end token)

        # [8, N] LongTensor (token ids e.g. 'dog' might be 1000)
        input_ids = batch_encoding.input_ids.to(self.device)
        # [8, N] FloatTensor, mask for attention use later. 1 for tokens to attend, 0 for padding tokens
        attention_mask = batch_encoding.attention_mask.to(self.device)

        # Text masking
        # Produce an [8, N] mask (True indicating tokens to mask)
        mask = self.mask(input_ids, batch_encoding.special_tokens_mask)

        # Don't mask the original input_ids as it is used for labels later
        # [8, N]
        masked_inputs = input_ids.clone().detach().to(self.device)
        # Mask token is 103 in BERT
        masked_inputs[mask] = 103

        # When BERT calculates loss with the labels, it does not compute loss for
        # elements with ID < 0. They use -100 as an example in the docs.
        # This ensures that all the un-masked outputs are not used
        # [8, N]
        text_labels = input_ids.clone().detach().to(self.device)
        text_labels[~mask] = -100

        # Useful for later
        text_length = input_ids.shape[1]

        # Get the embeddings for each token id
        # [8, N, 512]
        text_embedding = text_embedding_layer(masked_inputs)
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
        features_embedding = self.visual_projection(features)

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
        input_embeddings = torch.cat([text_embedding, features_embedding], dim=1)

        # Add another [SEP] token at the end to let BERT know the sequence is over
        # The SEP embedding was the last text token
        # Bit of a funky matrix access, but it ensures the second cat element is
        # [8, 1, 512] rather than just [8, 512].
        # Unsqueeze might be clearer in future
        # Resultant shape is [8, N+6, 512]
        expanded_embeddings = torch.cat(
            [
                input_embeddings,
                input_embeddings[:, (text_length - 1) : text_length, :],
            ],
            dim=1,
        )

        # Apply layer norm and droupout
        expanded_embeddings = self.embedding_layer_norm(expanded_embeddings)
        expanded_embeddings = self.embedding_dropout(expanded_embeddings)

        # Prep the labels for loss
        # Make sure the shape is right: [8, N+6]
        # otherwise BERT gets angry
        labels = torch.zeros(expanded_embeddings.shape[0:2]).to(self.device)
        # The first N tokens are the same as previously computed
        labels[:, 0:text_length] = text_labels
        # No prediction to be made on the vision tokens, so let it be -100
        labels[:, text_length:] = -100
        labels = labels.long()

        # Make one attention mask as the cat of:
        # Text attention mask, vision attention mask, and then the [SEP] element
        # [8, N+6]
        attention_mask = torch.cat(
            [
                attention_mask,
                vision_mask,
                torch.ones((attention_mask.shape[0], 1)).to(self.device),
            ],
            dim=1,
        )

        return expanded_embeddings, labels, attention_mask

    def forward(self, vision_features, vision_mask, caption=None):
        input_embeddings, labels, attention_mask = self.prepare_inputs(
            caption, vision_features, vision_mask
        )

        output = self.bert(
            inputs_embeds=input_embeddings,
            labels=labels,
            return_dict=True,
            attention_mask=attention_mask,
        )

        logits = output.logits
        loss = output.loss

        return logits, loss, labels

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
        caption, features, vision_mask = batch

        logits, loss, labels = self(features, vision_mask, caption)

        batch_targets = self.tokenizer.decode(labels[labels != -100].tolist())
        batch_preds = self.tokenizer.decode(
            torch.argmax(logits[labels != -100, :], dim=1).tolist()
        )

        # print(f"{batch_targets=}, {batch_preds=}")
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
    #     pass
