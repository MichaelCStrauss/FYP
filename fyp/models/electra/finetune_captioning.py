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
    AutoModel,
    AutoConfig,
)

from fyp.models.visualbert.model import VisualBERT
from fyp.models.electra.model import VisualElectra
from fyp.models.electra.config import VisualElectraConfig
from fyp.models.visualbert.config import VisualBERTConfig, TrainingObjective
from fyp.data.coco_captions import CocoCaptions
from fyp.models.utils.checkpoint_n_steps import CheckpointEveryNSteps
from pytorch_lightning.utilities.cloud_io import load as pl_load

work_dir = "models/electra"

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main(
    checkpoint: str,
    test: bool = False,
    overfit: float = 0,
    max_epochs: int = 1000,
):
    config: VisualElectraConfig = VisualElectraConfig()
    # Base BERT model
    config.tokenizer = AutoTokenizer.from_pretrained(
        "google/bert_uncased_L-4_H-512_A-8"
        # "bert-base-uncased"
    )
    gen_model_name = "google/bert_uncased_L-2_H-512_A-8"
    disc_model_name = "google/bert_uncased_L-8_H-768_A-12"
    config.hidden_size = 512

    gen_conf = AutoConfig.from_pretrained(gen_model_name)
    config.generator_model = AutoModelForMaskedLM.from_config(gen_conf)
    config.generator_hidden_size = 512

    disc_conf = AutoConfig.from_pretrained(disc_model_name)
    disc_conf.is_decoder = True
    config.discriminator_model = AutoModel.from_config(disc_conf)
    config.discriminator_hidden_size = 768

    full_model = VisualElectra.load_from_checkpoint(checkpoint, config=config)
    model = full_model.discriminator
    model.training_objective = TrainingObjective.Captioning
    model.add_lm_head()

    data = CocoCaptions()
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
            config={'checkpoint': checkpoint},
            tags=['electra-finetune']
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
