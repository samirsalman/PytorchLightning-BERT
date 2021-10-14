import os
import pandas as pd
import pytorch_lightning as pl
from src.pl_data.TextDataModule import TextDataModule
from src.pl_modules.BERTClassifier import BertTextClassifier
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    LABEL_COLUMN = cfg.data.label_column
    TEXT_COLUMN = cfg.data.text_column

    os.chdir(get_original_cwd())

    # PRE-PROCESSING AREA
    # read data
    train = pd.read_csv(
        os.path.join(cfg.data.data_dir, cfg.data.train_path), encoding="latin1"
    )
    test = pd.read_csv(
        os.path.join(cfg.data.data_dir, cfg.data.test_path), encoding="latin1"
    )

    # drop nan value in the columns: OriginalTweet and Sentiment
    train.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)
    test.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)

    # train validation splitting
    train, val = train_test_split(
        train, test_size=0.2, stratify=train[LABEL_COLUMN].values
    )

    train.to_csv(os.path.join(cfg.data.data_dir, cfg.data.train_path))
    val.to_csv(os.path.join(cfg.data.data_dir, cfg.data.val_path))
    test.to_csv(os.path.join(cfg.data.data_dir, cfg.data.test_path))

    print("Creating text datamodule")

    # DATA LOADING AREA
    text_datamodule = TextDataModule(
        data_dir=cfg.data.data_dir,
        bert_model=cfg.bert.model_name,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        max_len=cfg.bert.max_length,
        train_path=cfg.data.train_path,
        test_path=cfg.data.test_path,
        val_path=cfg.data.val_path,
        train_batch_size=cfg.data.batch_size,
    )
    print("Text datamodule created")

    # MODEL AREA
    print("Init the model")
    model = BertTextClassifier(
        bert_model=cfg.bert.model_name,
        label_column=LABEL_COLUMN,
        lr=cfg.optimizer.lr,
        n_classes=cfg.model.num_classes,
        n_training_steps=cfg.optimizer.n_training_steps,
        n_warmup_steps=cfg.optimizer.n_warmup_steps,
    )
    print("Model created")

    early_stopping = pl.callbacks.EarlyStopping(
        monitor=cfg.callbacks.early_stop.monitor,
        min_delta=cfg.callbacks.early_stop.min_delta,
        patience=cfg.callbacks.early_stop.patience,
        verbose=cfg.callbacks.early_stop.verbose,
        mode=cfg.callbacks.early_stop.mode,
    )

    print("Init the trainer")

    # TRAINING AREA
    trainer = pl.Trainer(
        callbacks=[early_stopping],
        deterministic=cfg.training.deterministic,
        gpus=1,
        max_epochs=cfg.training.max_epochs,
    )

    print("Starting the train")

    # start the train
    trainer.fit(model=model, datamodule=text_datamodule)


if __name__ == "__main__":
    main()
