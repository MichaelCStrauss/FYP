import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class SimpleRNNCaptioner(pl.LightningModule):
    def __init__(self, vocab_size: int, learning_rate: float = None):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.feature_count = 2048
        self.embedding_dim = 256

        self.features_dropout = nn.Dropout(0.0)
        self.features_dense = nn.Linear(self.feature_count, self.embedding_dim)

        self.embedding = nn.Embedding(vocab_size, 256)
        self.embedding_drop = nn.Dropout(0.5)
        self.text_encoder = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True,
        )

        self.decoder1 = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.decoder1_activation = nn.ReLU()
        self.decoder2 = nn.Linear(self.embedding_dim, vocab_size)

    # def setup(self, stage):
    #     self.logger.experiment.add_tags(["captioning", "simple-rnn", "flickr8k"])

    def forward(self, features, captions, lengths):

        features = self.features_dropout(features)
        features = self.features_dense(features)

        captions = self.embedding(captions)
        captions = self.embedding_drop(captions)

        captions, _ = self.text_encoder(captions)
        captions = captions[:, -1]

        decoder = torch.cat([captions, features], dim=1)

        decoder = self.decoder1(decoder)
        decoder = self.decoder1_activation(decoder)
        decoder = self.decoder2(decoder)

        return decoder

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=(self.learning_rate if self.learning_rate is not None else 5e-4),
        )

    def training_step(self, batch, batch_idx):
        features, captions, lengths, targets, _ = batch

        y_hat = self(features, captions, lengths)

        loss = F.cross_entropy(y_hat, targets)

        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        features, captions, lengths, targets, _ = batch

        y_hat = self(features, captions, lengths)

        loss = F.cross_entropy(y_hat, targets)

        return {"loss": loss, "log": {"val_loss": loss}}
