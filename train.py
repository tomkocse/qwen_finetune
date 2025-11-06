"""
This script fine-tunes Qwen/Qwen2-1.5B-Base on wmt/wmt17 with the chat template.
Note, this script has not been tested on multi-gpu setting.

References: https://huggingface.co/learn/llm-course/chapter3/3
https://huggingface.co/docs/optimum-neuron/en/training_tutorials/finetune_qwen3
"""
import copy
import logging
import os
from functools import partial
from typing import Any

from datasets import interleave_datasets
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForCausalLM
from transformers import Trainer, HfArgumentParser

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)

logger = logging.getLogger(__name__)

LANG_CODE_TO_NAME = {
    "en": "English",
    "zh": "Chinese"
}

MAX_SEQ_LEN = 512


def main():
    # Step 1: process dataset

    # ~2GB. `train`, `validation`, `test` subsets
    # only choose a few training samples, to reduce preprocessing time. It takes ~1min to preprocess 1M samples.
    train_data = load_dataset("wmt/wmt17", "zh-en")["train"].take(10_000)
    valid_data = load_dataset("wmt/wmt17", "zh-en")["validation"]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Base")

    def reformat_mt_data(example: dict, src_lang: str, tgt_lang: str) -> dict:
        """
        Reformat the data to chat template.
        """
        chats = []
        for ex in example["translation"]:
            src_sent = ex[src_lang]
            tgt_sent = ex[tgt_lang]
            chats.append(
                [
                    {
                        "role": "user",
                        "content": f"Translate this from {LANG_CODE_TO_NAME[src_lang]} to {LANG_CODE_TO_NAME[tgt_lang]}:\n{src_sent}"
                    },
                    {
                        "role": "assistant", "content": f"{tgt_sent}"
                    }
                ]
            )
        chats = tokenizer.apply_chat_template(
            chats, tokenize=False, add_generation_prompt=False
        )

        # do not need thinking tokens
        chats = [
            x.replace("\n<think>\n\n</think>\n", "")
            for x in chats
        ]

        return {"messages": chats}

    # combine two directions
    train_data = interleave_datasets([
        train_data.map(partial(reformat_mt_data, src_lang="en", tgt_lang="zh"), batched=True, batch_size=128,
                       remove_columns="translation"),
        train_data.map(partial(reformat_mt_data, src_lang="zh", tgt_lang="en"), batched=True, batch_size=128,
                       remove_columns="translation"),
    ]).shuffle(seed=42)
    valid_data = interleave_datasets([
        valid_data.map(partial(reformat_mt_data, src_lang="en", tgt_lang="zh"), batched=True, batch_size=128,
                       remove_columns="translation"),
        valid_data.map(partial(reformat_mt_data, src_lang="zh", tgt_lang="en"), batched=True, batch_size=128,
                       remove_columns="translation"),
    ])
    logger.info(f"One example:\n{next(iter(train_data))}")

    # tokenizer
    def tokenize_func(examples: dict[str, list[str]]):
        tokenized = tokenizer(
            examples["messages"],
            max_length=MAX_SEQ_LEN - 1,  # make room for EOS
            truncation=True,
        )
        ret = {
            "input_ids": [], "attention_mask": []
        }
        for inputs, att_mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
            if inputs[-1] != tokenizer.eos_token_id:  # 151643
                inputs.append(tokenizer.eos_token_id)
                att_mask.append(1)
            ret["input_ids"].append(inputs)
            ret["attention_mask"].append(att_mask)
        return ret

    train_data = train_data.map(
        tokenize_func, batched=True, batch_size=128, remove_columns="messages"
    )
    valid_data = valid_data.map(
        tokenize_func, batched=True, batch_size=128, remove_columns="messages"
    )
    logger.info(f"One example:\n{next(iter(train_data))}")

    # 2. Train
    # see https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer#transformers.TrainingArguments
    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collator(features: list[dict[str, Any]]):
        padded_res = data_collator(features)
        padded_res["labels"] = copy.deepcopy(padded_res["input_ids"])
        return padded_res

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == '__main__':
    main()
