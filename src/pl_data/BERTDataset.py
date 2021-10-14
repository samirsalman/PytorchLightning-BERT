from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch


class BERTDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_token_len: int = 128,
        label_column: str = "label",
        text_column: str = "text",
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_column = text_column
        self.max_token_len = max_token_len
        self.label_column = label_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row[self.text_column]
        labels = data_row[self.label_column]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels),
        )
