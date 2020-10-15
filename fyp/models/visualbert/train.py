# %%
import typer
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoConfig,
)

from fyp.models.visualbert.model import VisualBERT
from fyp.data.coco_captions import CocoCaptions
from fyp.models.visualbert.config import VisualBERTConfig, TrainingObjective

work_dir = "models/visualbert"

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main(
    training_objective: TrainingObjective,
    test: bool = False,
    overfit: float = 0,
    max_epochs: int = 1000,
):

    config: VisualBERTConfig = VisualBERTConfig()
    # Base BERT model
    config.tokenizer = AutoTokenizer.from_pretrained(
        "google/bert_uncased_L-4_H-512_A-8"
        # "bert-base-uncased"
    )
    config.training_objective = training_objective
    bert_model_name = "google/bert_uncased_L-4_H-512_A-8"
    config.hidden_size = 512

    if training_objective == TrainingObjective.MaskedLanguageModelling:
        config.bert_model = AutoModelForMaskedLM.from_pretrained(bert_model_name)
    elif training_objective == TrainingObjective.Captioning:
        c = AutoConfig.from_pretrained(bert_model_name)
        c.is_decoder = True
        config.bert_model = AutoModelForCausalLM.from_config(c)

    data = CocoCaptions()
    data.prepare_data()
    data.setup()

    model = VisualBERT(config)

    logger = None

    fast_dev_run = test & (overfit == 0)

    if test is not True:
        logger = WandbLogger(
            project="final-year-project",
            offline=False,
            log_model=True,
            save_dir=work_dir,
        )

    trainer = pl.Trainer(
        gpus=1,
        fast_dev_run=fast_dev_run,
        default_root_dir=work_dir,
        row_log_interval=10,
        logger=logger,
        max_epochs=max_epochs,
        overfit_batches=overfit,
        # check_val_every_n_epoch=1000 if overfit > 0 else 1,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    typer.run(main)
