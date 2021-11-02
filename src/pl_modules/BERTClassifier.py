import pytorch_lightning as pl
import torch.nn as nn
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from torchmetrics import Accuracy, F1
import torch
import numpy as np
import logging


class BertTextClassifier(pl.LightningModule):
    def __init__(
        self,
        bert_model: str,
        n_classes: int,
        lr: float = 2e-5,
        label_column: str = "label",
        n_training_steps=None,
        n_warmup_steps=None,
        logger=None,
    ):
        """Bert Classifier Model

        Args:
            bert_model (str): huggingface bert model
            n_classes (int): number of output classes
            lr (float, optional): learning rate value. Defaults to 2e-5.
            label_column (str, optional): the name of the label column in the dataframe. Defaults to "label".
            n_training_steps ([type], optional): optimizer parameter. Defaults to None.
            n_warmup_steps ([type], optional): optimizer parameter. Defaults to None.
        """

        super().__init__()
        self.bert_model = bert_model
        self.label_column = label_column
        self.bert = AutoModel.from_pretrained(bert_model, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_classes = n_classes
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.f1 = F1(num_classes=n_classes, average="macro")
        self.accuracy = Accuracy(num_classes=n_classes, average="macro")

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.long())
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        outputs = torch.argmax(outputs, dim=1)
        accuracy = self.accuracy(outputs, labels)
        f1 = self.f1(outputs, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", accuracy, prog_bar=True, logger=True)
        self.log("train_f1", f1, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        outputs = torch.argmax(outputs, dim=1)
        accuracy = self.accuracy(outputs, labels)
        f1 = self.f1(outputs, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy, prog_bar=True, logger=True)
        self.log("val_f1", f1, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        outputs = torch.argmax(outputs, dim=1)
        accuracy = self.accuracy(outputs, labels)
        f1 = self.f1(outputs, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", accuracy, prog_bar=True, logger=True)
        self.log("test_f1", f1, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )
        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )
