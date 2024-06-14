import pkg_resources
import json
import os

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm

from . import templates

DATA_DIR = pkg_resources.resource_filename("cs336_alignment", "data")


def template(data_dir=DATA_DIR):
    in_file = f"{data_dir}/hh_train.json"
    with open(in_file) as in_f:
        records = json.load(in_f)

    for rtype in ["chosen", "rejected"]:
        texts = []
        for record in records:
            prompt = record["prompt"]
            response = record[rtype]
            text = templates.alpaca_chat.format(prompt=prompt, response=response)
            # text = tokenizer.bos_token + text + tokenizer.eos_token
            text = "<|begin_of_text|>" + text + "<|end_of_text|>"
            texts.append(text)

        out_file = f"{data_dir}/{rtype}_templated.json"
        with open(out_file, "w") as out_f:
            json.dump(texts, out_f)


def tokenize(hf_model="meta-llama/Meta-Llama-3-8B", data_dir=DATA_DIR):
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer.pad_token = tokenizer.eos_token

    for rtype in ["chosen", "rejected"]:
        with open(f"{data_dir}/{rtype}_templated.json") as in_f:
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

        torch.save(input_ids, f"{data_dir}/input_ids_{rtype}.pt")
        torch.save(attention_mask, f"{data_dir}/attention_mask_{rtype}.pt")


def infer(
    rtype,
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

    input_ids = torch.load(f"{data_dir}/input_ids_{rtype}.pt", map_location="cpu")
    attention_mask = torch.load(
        f"{data_dir}/attention_mask_{rtype}.pt", map_location="cpu"
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

    torch.save(ll, f"{data_dir}/ll_{rtype}.pt")


def train(
    *,
    hf_model="/lfs/ampere2/0/ranjanr/cs336-a5/out/sft_single_gpu/hf_model",
    batch_size=1,
    val_size=256,
    lr=1e-6,
    weight_decay=0.01,
    beta=0.1,
    grad_acc_steps=2,
    opt="rmsprop",
    torch_compile=False,
    fsdp=False,
):
    torch.set_float32_matmul_precision("medium")

    if fsdp:
        init_process_group(backend="nccl")

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"

    # load data
    all_input_ids = {}
    all_attention_mask = {}
    all_ref_ll = {}
    for rtype in ["chosen", "rejected"]:
        all_input_ids[rtype] = torch.load(
            f"{DATA_DIR}/input_ids_{rtype}.pt", map_location="cpu"
        )
        all_attention_mask[rtype] = torch.load(
            f"{DATA_DIR}/attention_mask_{rtype}.pt", map_location="cpu"
        )
        all_ref_ll[rtype] = torch.load(f"{DATA_DIR}/ll_{rtype}.pt", map_location="cpu")

    # shuffle
    len_data = all_input_ids["chosen"].size(0)
    shuffle_idx = torch.randperm(len_data)
    for rtype in ["chosen", "rejected"]:
        all_input_ids[rtype] = all_input_ids[rtype][shuffle_idx]
        all_attention_mask[rtype] = all_attention_mask[rtype][shuffle_idx]
        all_ref_ll[rtype] = all_ref_ll[rtype][shuffle_idx]

    # drop last
    eff_batch_size = batch_size * grad_acc_steps
    len_drop = len_data % eff_batch_size
    if len_drop > 0:
        for rtype in ["chosen", "rejected"]:
            all_input_ids[rtype] = all_input_ids[rtype][:-len_drop]
            all_attention_mask[rtype] = all_attention_mask[rtype][:-len_drop]
            all_ref_ll[rtype] = all_ref_ll[rtype][:-len_drop]
    len_data = all_input_ids["chosen"].size(0)

    # split
    assert val_size % eff_batch_size == 0
    split_input_ids = {}
    split_attention_mask = {}
    split_ref_ll = {}
    for rtype in ["chosen", "rejected"]:
        train_split, val_split = torch.split(
            all_input_ids[rtype],
            [len_data - val_size, val_size],
        )
        split_input_ids[rtype] = {"train": train_split, "val": val_split}

        train_split, val_split = torch.split(
            all_attention_mask[rtype],
            [len_data - val_size, val_size],
        )
        split_attention_mask[rtype] = {"train": train_split, "val": val_split}

        train_split, val_split = torch.split(
            all_ref_ll[rtype],
            [len_data - val_size, val_size],
        )
        split_ref_ll[rtype] = {"train": train_split, "val": val_split}

    # model
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    if fsdp:
        model = FSDP(model, device_id=device)
    else:
        model = model.to(device)

    if torch_compile:
        model = torch.compile(model)

    # optimizer
    if opt == "adamw":
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == "rmsprop":
        opt = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == "sgd":
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    lrs = optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        total_steps=len_data // eff_batch_size,
        pct_start=0.03,
        anneal_strategy="cos",
        final_div_factor=10,
    )

    # train one epoch
    num_batches = len_data // batch_size
    for batch_idx in tqdm(range(num_batches)):
        model.train()

        global_batch_idx = batch_idx * world_size + rank
        global_batch_size = batch_size * world_size
        begin_idx = global_batch_idx * global_batch_size
        end_idx = begin_idx + batch_size
        idx = torch.arange(begin_idx, end_idx)
        input_ids = {}
        attention_mask = {}
        ref_ll = {}
        for rtype in ["chosen", "rejected"]:
            input_ids[rtype] = split_input_ids[rtype]["train"][idx]
            input_ids[rtype] = input_ids[rtype]
            attention_mask[rtype] = split_attention_mask[rtype]["train"][idx]
            attention_mask[rtype] = attention_mask[rtype]
            ref_ll[rtype] = split_ref_ll[rtype]["train"][idx]
            ref_ll[rtype] = ref_ll[rtype]

        input_ids = torch.cat([input_ids["chosen"], input_ids["rejected"]])
        input_ids = input_ids.to(device)
        attention_mask = torch.cat(
            [attention_mask["chosen"], attention_mask["rejected"]]
        )
        attention_mask = attention_mask.to(device)
        ref_ll = torch.cat([ref_ll["chosen"], ref_ll["rejected"]])
        rel_ll = ref_ll.to(device)

        logits = model(
            input_ids,
            attention_mask=attention_mask,
        ).logits
        token_ll = -F.cross_entropy(
            logits[:, :-1].transpose(1, 2),
            input_ids[:, 1:],
            reduction="none",
        )
        ll = token_ll.sum(-1)

        lm_log_ratio = ll[:batch_size] - ll[batch_size:]
        ref_log_ratio = ref_ll[:batch_size] - ref_ll[batch_size:]
        ref_log_ratio = ref_log_ratio.to(device)
        diff = lm_log_ratio - ref_log_ratio
        loss = -F.logsigmoid(-beta * diff)
        loss = loss / grad_acc_steps
        # if rank == 0:
        #     print(loss.item())

        loss.backward()

        if (batch_idx + 1) % grad_acc_steps == 0:
            torch.cuda.empty_cache()
            opt.step()
            lrs.step()
            opt.zero_grad(set_to_none=True)

    if fsdp:
        destroy_process_group()
