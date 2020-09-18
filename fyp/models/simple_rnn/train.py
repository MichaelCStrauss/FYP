# %%
from dotenv import load_dotenv
import typer

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


from fyp.models.simple_rnn.model import SimpleRNNCaptioner
from fyp.data.flickr8k import Flickr8kDataModule

load_dotenv()

work_dir = "models/simple_rnn"


def main(test: bool = False, max_epochs: int = None):
    data = Flickr8kDataModule()
    data.prepare_data()
    data.setup()

    model = SimpleRNNCaptioner(data.encoder.vocab_size)

    logger = None
    if test is not True:
        logger = WandbLogger(
            project="final-year-project", offline=False, log_model=True
        )

    trainer = pl.Trainer(
        gpus=1,
        fast_dev_run=test,
        default_root_dir=work_dir,
        row_log_interval=10,
        logger=logger,
        max_epochs=max_epochs,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    typer.run(main)
