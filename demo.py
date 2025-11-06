"""
This is an example to interact with the model for translation.
The base model has pretty good performance, but
1. It does not work when applying the chat template;
2. Still some wrong translations. e.g., try to translate this into Chinese:
Of course, the fall of the house of Lehman Brothers has nothing to do with the fall of the Berlin Wall.
"""
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="Qwen/Qwen2-1.5B",
        help="Either the pre-trained Qwen model, or the dir to your fine-tuned checkpoint."
    )
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False
    )
    parser.add_argument(
        "--src_lang", type=str, default="Chinese"
    )
    parser.add_argument(
        "--tgt_lang", type=str, default="English"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    device = "cuda:0"

    # won't modify the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)

    while True:
        input_sent = input(f"(Input `quit` to quit) Input a {src_lang} sentence:\n").strip()
        if input_sent.lower() == "quit":
            break

        if args.apply_chat_template:
            prompt = f"Translate this from {src_lang} to {tgt_lang}:\n{input_sent}"
            messages = [
                {"role": "user",
                 "content": prompt},
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            # Print this to see the effect of chat template:
            # inputs = tokenizer.apply_chat_template(
            #     messages,
            #     add_generation_prompt=True,
            #     tokenize=False,
            #     return_dict=False,
            # )
        else:
            prompt = f"Translate this from {src_lang} to {tgt_lang}:\n{src_lang}: {input_sent}\n{tgt_lang}:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(**inputs, max_new_tokens=128)
        print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))


if __name__ == '__main__':
    main()
