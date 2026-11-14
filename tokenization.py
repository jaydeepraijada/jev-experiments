import os 

import torch 
from transformers import AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false") # the tokenizer is shared with DataLoader worker processes


model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_prompt(question,choices):
    choice_begin_token = tokenizer.encode("<option>")
    choice_end_token = tokenizer.encode("</option>")
    answer_prompt = "The answer is: "

    question_tokens = tokenizer.encode(question)
    answer_tokens = tokenizer.encode(answer_prompt)

    choice_tokens = [
        tokenizer.encode(choice) for choice in choices
    ]

    final_tokens = []
    final_tokens.extend(question_tokens)

    span = {
        "question_end_idx": len(question_tokens) - 1,
        "choices_end_idx": [],
        "answer_end_idx": 0
    }

    for choice in choice_tokens:
        final_tokens.extend(choice_begin_token)
        final_tokens.extend(choice)
        final_tokens.extend(choice_end_token)
        span["choices_end_idx"].append(len(final_tokens) - 1)

    final_tokens.extend(answer_tokens)
    span["answer_end_idx"] = len(final_tokens) - 1

    return final_tokens, span
