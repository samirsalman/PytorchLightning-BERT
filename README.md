# PytorchLightning-BERT

<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/static/v1?label=code&color=blueviolet&logo=pytorchlightning&message=PytorchLightning"></a>
    <a href="https://hydra.cc/"><img alt="Conf: hydra" src="https://img.shields.io/badge/conf-hydra-blue"></a>
    <a href="https://www.docker.com/"><img alt="Dockerfile" src="https://img.shields.io/static/v1?label=Dockerfile&color=blue&logo=docker&message=available"></a>
</p>

PytorchLigtning BERT is an implementation of BERT for classification tasks using PytorchLightning. PL-BERT is composed of 3 different parts:
- Configs: Hydra configuration about data, training, and bert model
- Datamodule: PytorchLigtning datamodule that abstract textual data preprocessing step. You can use it with all kinds of textual data for classification tasks.
- Classifier: PytorchLigtning module that implements a BERT-based classifier. 



## Hydra Configuration

You can reuse this template in any text classification task, changing hydra configurations. In the conf file, you can find some directories, each of them contains a configuration of a specific component. For example, bert directory contains the ```bert.yaml``` file, which specifies the configuration of the bert model used in your experiments. By default, it contains:
```yaml
model_name: bert-base-cased
max_length: 120
```

## Datamodule

## Model


## Run the project

### Pre-run steps
- Change hydra configuration, based on experiment purposes
- Check data
- Check model

### With Docker
From the project directory open the cmd and launch the following commands:

- ```docker build --tag pl_bert```

- ```docker run pl_bert -a```

