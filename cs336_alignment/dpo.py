import torch
import torch.nn.functional as F

from . import templates


def tokenize(tokenizer, prompt, response):
    text = templates.alpaca_chat.format(prompt=prompt, response=response)
    text += tokenizer.eos_token
    tokens = tokenizer(text, return_tensors="pt").input_ids
    return tokens


def log_prob(lm, tokens):
    return lm(tokens, labels=tokens).loss * (tokens.size(1) - 1)


def log_ratio(lm, chosen_tokens, rejected_tokens):
    chosen_log_prob = log_prob(lm, chosen_tokens)
    rejected_log_prob = log_prob(lm, rejected_tokens)
    return chosen_log_prob - rejected_log_prob


def dpo_loss(
    lm,
    lm_ref,
    tokenizer,
    beta,
    prompt,
    response_chosen,
    response_rejected,
):
    chosen_tokens = tokenize(tokenizer, prompt, response_chosen)
    rejected_tokens = tokenize(tokenizer, prompt, response_rejected)
    lm_log_ratio = log_ratio(lm, chosen_tokens, rejected_tokens)
    lm_ref_log_ratio = log_ratio(lm_ref, chosen_tokens, rejected_tokens)
    diff = lm_log_ratio - lm_ref_log_ratio
    loss = -F.logsigmoid(-beta * diff)
    return loss
