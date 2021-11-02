# PytorchLightning-BERT

<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/static/v1?label=code&color=blueviolet&logo=pytorchlightning&message=PytorchLightning"></a>
    <a href="https://hydra.cc/"><img alt="Conf: hydra" src="https://img.shields.io/badge/conf-hydra-blue"></a>
    <a href="https://www.docker.com/"><img alt="Dockerfile" src="https://img.shields.io/static/v1?label=Dockerfile&color=blue&logo=docker&message=available"></a>
</p>

PytorchLigtning BERT is a **reusable** implementation of BERT for classification tasks using PytorchLightning. PL-BERT is composed of 3 different parts:
- **Hydra Configuration**: Hydra configuration about data, training, and bert model
- **Datamodule**: PytorchLigtning datamodule that abstract textual data preprocessing step. You can use it with all kinds of textual data for classification tasks.
- **Model**: PytorchLigtning module that implements a BERT-based classifier. 



## Hydra Configuration

You can reuse this template in any text classification task, changing hydra configurations. In the conf directory, you can find some directories, each of them contains a configuration of a specific component. For example, bert directory contains the ```bert.yaml``` file, which specifies the configuration of the bert model used in your experiments. By default, it contains:
```yaml
model_name: bert-base-cased
max_length: 120
```

## Datamodule
```PytorchLightning-BERT/src/pl_data/TextDataModule.py```

The datamodule component contains all data logic. It take in input the textual dataset, converts string labels in integer labels and create the train, test and optionally validation dataloaders. The input arguments of TextDataModule are:
```python
train_path: str,
test_path: str,
val_path: str,
bert_model: str,
text_column: str = "text",
data_dir: str = "data/",
label_column: str = "label",
train_batch_size: int = 32,
max_len: int = 120,
```

Each of that arguments is given by [data and bert] config files, which you can find in the confi directory.


## Model
```PytorchLightning-BERT/src/pl_modules/BERTClassifier.py```

The model is a PytorchLightning module that contains the model implementation. In our project we implement a BERT-based classifier, you can create any model for your specific task. The input argument of BERTClassifier.py are:
```python
bert_model: str,
n_classes: int,
lr: float = 2e-5,
label_column: str = "label",
n_training_steps=None,
n_warmup_steps=None,
```
Each of that arguments is given by [training, data, model and bert] config files, which you can find in the confi directory.
By default the BERT classifier uses the Accuracy and F1 metrics.

## Run the project

### Pre-run steps
- Change hydra configuration, based on experiment purposes like dicussed in Hydra Configuration section.
- Check data
- Check model

### Local or Cloud environment
You can install all dependencies running the following command from the project root directory:
```pip install -r requirements.txt```

When all requirements have been installed, you can run the training of the model launching the ```main.py``` script.

### With Docker
From the project directory open the cmd and launch the following commands:

- ```docker build --tag pl_bert```

- ```docker run pl_bert -a```

