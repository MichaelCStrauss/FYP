# %%
import os
from dotenv import load_dotenv

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from fyp.models.simple_rnn.model import SimpleRNNCaptioner
from fyp.data.flickr8k import Flickr8kDataModule

load_dotenv()

work_dir = "models/simple_rnn"


# %%
data = Flickr8kDataModule()
data.prepare_data()
data.setup()

model = SimpleRNNCaptioner(data.encoder.vocab_size)

test = False

logger = None
if test is not True:
    logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        workspace=os.environ.get("COMET_WORKSPACE"),
        save_dir=work_dir,
        project_name=os.environ.get("COMET_PROJECT_NAME"),
    )

# %%
trainer = pl.Trainer(
    gpus=1,
    default_root_dir=work_dir,
    row_log_interval=10,
    logger=logger,
    auto_lr_find=True,
)
trainer.fit(model, data)
