import pandas as pd
import pytorch_lightning as pl
from src.pl_data.TextDataModule import TextDataModule
from src.pl_modules.BERTClassifier import BertTextClassifier
from sklearn.model_selection import train_test_split

LABEL_COLUMN = "Sentiment"
TEXT_COLUMN = "OriginalTweet"

if __name__ == "__main__":
    train = pd.read_csv("data/Corona_NLP_train.csv", encoding="latin1")
    test = pd.read_csv("data/Corona_NLP_test.csv", encoding="latin1")

    train.dropna(subset=["OriginalTweet", "Sentiment"], inplace=True)
    test.dropna(subset=["OriginalTweet", "Sentiment"], inplace=True)

    train, val = train_test_split(
        train, test_size=0.2, stratify=train[LABEL_COLUMN].values
    )
    train.to_csv("data/Corona_NLP_train.csv")
    val.to_csv("data/Corona_NLP_val.csv")
    val.to_csv("data/Corona_NLP_test.csv")

    print("Creating text datamodule")

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
    print("Text datamodule created")

    print("Init the model")
    model = BertTextClassifier(
        bert_model="bert-base-cased",
        label_column=LABEL_COLUMN,
        lr=2e-5,
        n_classes=2,
    )
    print("Model created")

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="min",
    )

    print("Init the trainer")

    trainer = pl.Trainer(
        callbacks=[early_stopping],
        deterministic=True,
        gpus=1,
        max_epochs=10,
    )

    print("Starting the train")

    trainer.fit(model=model, datamodule=text_datamodule)
