import pkg_resources
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm

from . import templates

DATA_DIR = pkg_resources.resource_filename("cs336_alignment", "data")


def template(data_dir=DATA_DIR):
    in_file = f"{data_dir}/hh_train.json"
    with open(in_file) as in_f:
        records = json.load(in_f)

    for response_type in ["chosen", "rejected"]:
        texts = []
        for record in records:
            prompt = record["prompt"]
            response = record[response_type]
            text = templates.alpaca_chat.format(prompt=prompt, response=response)
            # text = tokenizer.bos_token + text + tokenizer.eos_token
            text = "<|begin_of_text|>" + text + "<|end_of_text|>"
            texts.append(text)

        out_file = f"{data_dir}/{response_type}_templated.json"
        with open(out_file, "w") as out_f:
            json.dump(texts, out_f)


def tokenize(hf_model="meta-llama/Meta-Llama-3-8B", data_dir=DATA_DIR):
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer.pad_token = tokenizer.eos_token

    for response_type in ["chosen", "rejected"]:
        with open(f"{data_dir}/{response_type}_templated.json") as in_f:
            texts = json.load(in_f)

        encoding = tokenizer(
            texts,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_attention_mask=True,
            verbose=True,
        )
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask

        torch.save(input_ids, f"{data_dir}/input_ids_{response_type}.pt")
        torch.save(attention_mask, f"{data_dir}/attention_mask_{response_type}.pt")


def infer(
    response_type,
    batch_size=64,
    hf_model="meta-llama/Meta-Llama-3-8B",
    data_dir=DATA_DIR,
):
    device = "cuda"
    torch.set_float32_matmul_precision("medium")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device)
    model = torch.compile(model)
    model.eval()

    input_ids = torch.load(
        f"{data_dir}/input_ids_{response_type}.pt", map_location="cpu"
    )
    attention_mask = torch.load(
        f"{data_dir}/attention_mask_{response_type}.pt", map_location="cpu"
    )

    with torch.no_grad():
        lls = []
        for i in tqdm(range(0, input_ids.size(0), batch_size)):
            batch_input_ids = input_ids[i : i + batch_size]
            batch_attention_mask = attention_mask[i : i + batch_size]
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
            token_ll = -F.cross_entropy(
                outputs.logits[:, :-1].transpose(1, 2),
                batch_input_ids[:, 1:],
                reduction="none",
            )
            ll = token_ll.sum(-1)
            lls.append(ll)
        ll = torch.cat(lls)

    torch.save(ll, f"{data_dir}/ll_{response_type}.pt")


def vllm_log_prob(
    response_type,
    hf_model="meta-llama/Meta-Llama-3-8B",
    data_dir=DATA_DIR,
):
    with open(f"{data_dir}/{response_type}_templated.json") as in_f:
        texts = json.load(in_f)

    lm = LLM(model=hf_model)
    sampling_params = SamplingParams(
        max_tokens=1,
        prompt_logprobs=1,
    )

    outputs = lm.generate(texts, sampling_params)
    breakpoint()
