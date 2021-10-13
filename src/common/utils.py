import pytorch_lightning as pl


def classify_single_example(
    model: pl.LightningModule,
    sentence: str,
    tokenizer: AutoTokenizer,
    max_token_len: int = 128,
):

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
