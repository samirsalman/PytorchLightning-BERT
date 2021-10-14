from typing import Any
import pytorch_lightning as pl
from transformers import AutoTokenizer


def classify_single_example(
    model: pl.LightningModule,
    sentence: str,
    tokenizer: AutoTokenizer,
    max_token_len: int = 128,
) -> Any:
    """Predict a single example (model wrapper)

    Args:
        model (pl.LightningModule): trained model
        sentence (str): input sentence
        tokenizer (AutoTokenizer): same tokenizer used during the training
        max_token_len (int, optional): max lenght of the input tokens. Defaults to 128.

    Returns:
        Any: prediction
    """

    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_token_len,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    pred = model.predict(input_ids, attention_mask)
    return pred
