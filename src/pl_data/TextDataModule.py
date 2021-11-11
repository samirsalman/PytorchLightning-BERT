from typing import Optional
from src.pl_data.BERTDataset import BERTDataset
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import os
import pandas as pd
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import json


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        val_path: str,
        bert_model: str,
        text_column: str = "text",
        data_dir: str = "data/",
        label_column: str = "label",
        train_batch_size: int = 32,
        max_len: int = 120,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path

        self.train_data = pd.read_csv(
            os.path.join(data_dir, train_path), encoding="latin1"
        )
        self.test_data = pd.read_csv(
            os.path.join(data_dir, test_path), encoding="latin1"
        )
        self.val_data = pd.read_csv(os.path.join(data_dir, val_path), encoding="latin1")

        self.train_batch_size = train_batch_size

        self.bert_model = bert_model
        self.max_len = max_len
        self.label_column = label_column
        self.text_column = text_column

    def prepare_data(self):
        self.labelencoder = LabelEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_data["label"] = self.labelencoder.fit_transform(
                self.train_data[self.label_column].values
            )

            self.train_dataset = BERTDataset(
                data=self.train_data,
                tokenizer=self.tokenizer,
                max_token_len=self.max_len,
                text_column=self.text_column,
                label_column="label",
            )

            self.val_data["label"] = self.labelencoder.transform(
                self.val_data[self.label_column].values
            )
            self.val_dataset = BERTDataset(
                data=self.val_data,
                tokenizer=self.tokenizer,
                max_token_len=self.max_len,
                text_column=self.text_column,
                label_column="label",
            )

            encodings = dict(
                zip(self.labelencoder.classes_, range(len(self.labelencoder.classes_)))
            )

            with open("outputs/labelencoder.json", "w") as le:
                le.write(json.dumps(encodings))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data["label"] = self.labelencoder.transform(
                self.test_data[self.label_column].values
            )
            self.test_dataset = BERTDataset(
                data=self.test_data,
                tokenizer=self.tokenizer,
                max_token_len=self.max_len,
                text_column=self.text_column,
                label_column="label",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)
