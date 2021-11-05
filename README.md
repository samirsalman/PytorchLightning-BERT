# PytorchLightning-BERT

<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/static/v1?label=code&color=blueviolet&logo=pytorchlightning&message=PytorchLightning"></a>
    <a href="https://hydra.cc/"><img alt="Conf: hydra" src="https://img.shields.io/badge/conf-hydra-blue"></a>
    <a href="https://www.docker.com/"><img alt="Dockerfile" src="https://img.shields.io/static/v1?label=Dockerfile&color=blue&logo=docker&message=available"></a>
</p>

PytorchLigtning BERT is a **modular, tiny and reusable** implementation of BERT for classification tasks using PytorchLightning. PL-BERT is composed of 3 parts:
- **Hydra Configuration**: Hydra configuration about data, training, and bert model
- **Datamodule**: PytorchLigtning datamodule that abstract textual data preprocessing step. You can use it with all kinds of textual data for classification tasks.
- **Model**: PytorchLigtning module that implements a BERT-based classifier. 



## Hydra Configuration

You can use this template in any text classification task, just changing hydra configurations. In the config directory, you can find the yaml configuration files of each specific component. For example, bert directory contains the ```bert.yaml``` file, which specifies the configuration of the bert model used in your experiments. By default, it contains:
```yaml
model_name: bert-base-cased
max_length: 120
```
The 

## Datamodule
```PytorchLightning-BERT/src/pl_data/TextDataModule.py```

The datamodule component contains all torch data logic. It take in input the textual dataset (BERTDataset.py), converts string labels in integer labels and create the train, test and optionally validation dataloaders. The input arguments of TextDataModule are:
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

Each argument is given by [data and bert] config files, which you can find in the config directory.


## Model
```PytorchLightning-BERT/src/pl_modules/BERTClassifier.py```

The model is a PytorchLightning module that contains the model implementation. In our project we implement a BERT-based classifier, you can create any model for your specific task. The input arguments of BERTClassifier.py are:
```python
bert_model: str,
n_classes: int,
lr: float = 2e-5,
label_column: str = "label",
n_training_steps=None,
n_warmup_steps=None,
```
Each argument is given by [training, data, model and bert] config files, which you can find in the config directory.

## Add new metric

By default the BERT classifier uses the Accuracy and F1 metrics, but you can add any kind of metric from the **torchmetric package**. To add a new metric you can add it in the __init__ of BERTClassifier.py, like already done for F1 and Accuracy:
```python
from torchmetrics import Accuracy, F1

 class BertTextClassifier(pl.LightningModule):  
 
   def __init__(
        self,
        bert_model: str,
        n_classes: int,
        lr: float = 2e-5,
        label_column: str = "label",
        n_training_steps=None,
        n_warmup_steps=None,
    ):
        super().__init__()
        ...
        # init F1 function
        self.f1 = F1(num_classes=n_classes, average="macro")
        
        # init Accuracy function
        self.accuracy = Accuracy(num_classes=n_classes, average="macro")
```

And to compute and log the metrics you should add the metric computation in the training_step, validation_step and test_step methods. 

```python
def training_step(self, batch, batch_idx):
    ...
    #loss, outputs = self(input_ids, attention_mask, labels)
    #outputs = torch.argmax(outputs, dim=1)
    
    # compute accuracy using self.accuracy function defined in the __init__
    accuracy = self.accuracy(outputs, labels)
    
    # compute f1 using self.f1 function defined in the __init__
    f1 = self.f1(outputs, labels)
    
    #self.log("train_loss", loss, prog_bar=True, logger=True)
    self.log("train_accuracy", accuracy, prog_bar=True, logger=True)
    self.log("train_f1", f1, prog_bar=True, logger=True)
    #return {"loss": loss, "predictions": outputs, "labels": labels}
```

## Run the project

### Pre-run steps
- Change hydra configuration, based on experiment purposes like dicussed in Hydra Configuration section.
- Check data
- Check model
- Run main.py using: ```python main.py```

**Hydra allow you to override configuration settings using cmd arguments, for example you can use a different bert model and a different learning rate value using the following bash command:**
```bash 
python main.py --bert.model_name=roberta-large --optimizer.lr=3e-5
```

### Local or Cloud environment
You can install all dependencies running the following command from the project root directory:
```pip install -r requirements.txt```

When all requirements have been installed, you can run the training of the model launching the ```main.py``` script.

### With Docker
From the project directory open the cmd and launch the following commands:

- ```docker build --tag pl_bert```

- ```docker run pl_bert -a```

