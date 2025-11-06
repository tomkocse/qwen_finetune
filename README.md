# Qwen Fine-tuning


A simple example of fine-tuning Qwen-base on WMT En-Zh data.
The base model can do MT already (try `demo.py` _without_ `--apply_chat_template`), but 
(1) it does not use the chat template (try `demo.py` _with_ `--apply_chat_template`); (2) occasionally makes mistakes. 
The goal of this fine-tuning is mainly learning the chat template.
The MT performance should also be improved.

## Environments
I use python=3.11 and the latest Huggingface, Pytorch versions.
```shell
conda create -n qwen_ft python=3.11
conda activate qwen_ft
pip install torch==2.9.0 transformers==4.57.1 datasets==4.3.0 accelerate==1.11.0
# for visualizing train/dev loss curves
pip install tensorboardX==2.6.4 tensorboard==2.20.0
```

## Files
- `./env.sh`: In order to run on SLAI cluster, it is better to set up the environment.
The pre-trained model checkpoint and the dataset will be saved to `HF_HOME`.
So, ideally, everyone could share one copy of pre-trained model and data. Saves disk space bit.
```shell
conda activate xxx
export HF_HOME=xxx
```
- `demo.py`: for interaction. Load either the pre-trained model or the fine-tuned model, and 
let the model translate an input sentence.
- `train.py`: the actual fine-tuning code. Use `run.sh` to launch it.
