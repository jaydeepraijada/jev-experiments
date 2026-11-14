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

def generate_position_ids(span):
    position_ids = []

    start_idx = 0
    for i in range(span["question_end_idx"] + 1):
        position_ids.append(start_idx)
        start_idx += 1

    next_choice_begin_idx = start_idx

    for choice_end_idx in span["choices_end_idx"]:
        for idx in range(choice_end_idx - next_choice_begin_idx + 1):
            position_ids.append(start_idx + idx)

        next_choice_begin_idx = choice_end_idx + 1

    # Answer tokens start right after the longest choice, so every choice token is in their past
    answer_start_idx = max(position_ids) + 1
    for idx in range(span["answer_end_idx"] + 1 - len(position_ids)):
        position_ids.append(answer_start_idx + idx)

    return position_ids