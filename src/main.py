import pandas as pd
import pytorch_lighting as pl
from pl_data.TextDataModule import TextDataModule
from pl_modules.BERTClassifier import BertTextClassifier
from sklearn.model_selection import train_test_split

LABEL_COLUMN = "Sentiment"
TEXT_COLUMN = "OriginalTweet"

if __name__ == "__main__":
    train = pd.read_csv("data/Corona_NLP_train.csv")
    train, val = train_test_split(
        train, test_size=0.2, stratify=train[LABEL_COLUMN].values
    )
    train.to_csv("data/Corona_NLP_train.csv")
    val.to_csv("data/Corona_NLP_val.csv")

    text_datamodule = TextDataModule(
        data_dir="data/",
        bert_model="bert-base-cased",
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        max_len=120,
        train_path="Corona_NLP_train.csv",
        test_path="Corona_NLP_test.csv",
        val_path="Corona_NLP_val.csv",
        train_batch_size=32,
    )

    model = BertTextClassifier(
        bert_model="bert-base-cased",
        label_column=LABEL_COLUMN,
        lr=2e-5,
        n_classes=2,
    )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="min",
    )

    trainer = pl.Trainer(
        callbacks=[early_stopping],
        deterministic=True,
        gpus=1,
        enable_progress_bar=True,
        max_epochs=10,
    )

    trainer.fit(model=model, datamodule=text_datamodule)
