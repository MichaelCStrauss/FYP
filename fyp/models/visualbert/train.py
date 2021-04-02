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
from fyp.data.conceptual_captions import ConceptualCaptions
from fyp.models.visualbert.config import VisualBERTConfig, TrainingObjective
from fyp.models.utils.checkpoint_n_steps import CheckpointEveryNSteps

work_dir = "models/visualbert"

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main(
    training_objective: TrainingObjective,
    test: bool = False,
    overfit: float = 0,
    max_epochs: int = 1000,
    checkpoint: str = None,
):

    config: VisualBERTConfig = VisualBERTConfig()
    # Base BERT model
    config.tokenizer = AutoTokenizer.from_pretrained(
        "google/bert_uncased_L-4_H-512_A-8"
        # "bert-base-uncased"
    )
    config.training_objective = training_objective
    bert_model_name = "google/bert_uncased_L-8_H-768_A-12"
    config.hidden_size = 768

    if training_objective == TrainingObjective.MaskedLanguageModelling:
        config.bert_model = AutoModelForMaskedLM.from_pretrained(bert_model_name)
    elif training_objective == TrainingObjective.Captioning:
        c = AutoConfig.from_pretrained(bert_model_name)
        c.is_decoder = True
        config.bert_model = AutoModelForCausalLM.from_config(c)

    if checkpoint is None:
        model = VisualBERT(config)
    else:
        model = VisualBERT.load_from_checkpoint(checkpoint, config=config)

    data = ConceptualCaptions()
    data.prepare_data()
    data.setup()

    logger = None

    fast_dev_run = test & (overfit == 0)

    if test is not True:
        logger = WandbLogger(
            project="final-year-project",
            offline=False,
            log_model=True,
            save_dir=work_dir,
        )

    callbacks = [CheckpointEveryNSteps(50000)]

    trainer = pl.Trainer(
        gpus=1,
        fast_dev_run=fast_dev_run,
        default_root_dir=work_dir,
        log_every_n_steps=10,
        logger=logger,
        max_epochs=max_epochs,
        overfit_batches=overfit,
        callbacks=callbacks
        # check_val_every_n_epoch=1000 if overfit > 0 else 1,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    typer.run(main)
