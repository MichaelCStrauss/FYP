# %%
import typer
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


from fyp.models.visualbert.model import VisualBERT
from fyp.data.flickr8k import Flickr8kDataModule, DatasetType

work_dir = "models/visualbert"

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main(test: bool = False, overfit: float = 0, max_epochs: int = 1000):
    data = Flickr8kDataModule(DatasetType.MaskedLanguageModel)
    data.prepare_data()
    data.setup()

    model = VisualBERT()

    logger = None

    fast_dev_run = test & (overfit == 0)

    if test is not True:
        logger = WandbLogger(
            project="final-year-project", offline=False, log_model=True, save_dir=work_dir
        )

    trainer = pl.Trainer(
        gpus=1,
        fast_dev_run=fast_dev_run,
        default_root_dir=work_dir,
        row_log_interval=10,
        logger=logger,
        max_epochs=max_epochs,
        checkpoint_callback=None,
        overfit_batches=overfit,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    typer.run(main)
