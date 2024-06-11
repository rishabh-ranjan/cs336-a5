import json
import os
from pathlib import Path
import pkg_resources
import random
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn, optim
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import wandb

from . import templates


def tokenize(hf_model, in_file, out_file):
    with open(in_file, "r") as f:
        lines = f.readlines()

    docs = []
    for line in lines:
        record = json.loads(line)
        doc = templates.alpaca_chat.format(**record)
        docs.append(doc)

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokens_list = []
    for doc in tqdm(docs):
        text = "<|begin_of_text|>" + doc + "<|end_of_text|>"
        tokens = tokenizer(text, add_special_tokens=False).input_ids
        tokens_list.append(tokens)

    random.shuffle(tokens_list)
    tokens = []
    for t in tokens_list:
        tokens += t
    data = torch.tensor(tokens)
    torch.save(data, out_file)


def main(
    *,
    hf_model="meta-llama/Meta-Llama-3-8B",
    ctx_len=512,
    batch_size=2,
    grad_acc_steps=16,
    lr=2e-5,
    weight_decay=0.01,
    wandb_project="sft",
    grad_clip=1.0,
    torch_compile=True,
    save_every_n_batches=8_000,
    out_dir="out/sft_multi_gpu",
    split="train",
):
    torch.backends.cudnn.benchmark = True

    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        is_master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_world_size = 1
        device = "cuda"
        is_master_process = True

    if is_master_process:
        if wandb_project:
            wandb.init(project=wandb_project, config=locals())

    data_file = pkg_resources.resource_filename(
        "cs336_alignment", f"data/sft/{split}.pt"
    )
    data = torch.load(data_file)
    data = data.to(device)
    num_batches = (data.size(0) - 1) // ctx_len // ddp_world_size
    num_steps = num_batches // grad_acc_steps
    num_batch_tokens = batch_size * ctx_len

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device)

    if torch_compile:
        torch.set_float32_matmul_precision("medium")
        model = torch.compile(model)

    if is_ddp:
        model = FSDP(model, device_id=ddp_local_rank)

    wd_params = []
    non_wd_params = []
    for param in model.parameters():
        if param.dim() >= 2:
            wd_params.append(param)
        else:
            non_wd_params.append(param)
    opt = optim.AdamW(
        [
            {"params": wd_params, "weight_decay": weight_decay},
            {"params": non_wd_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    lrs = optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        total_steps=num_steps,
        pct_start=0.03,
        anneal_strategy="cos",
        final_div_factor=10,
    )

    def checkpoint():
        tic = time.time()
        try:
            Path(f"{out_dir}/model.pt").rename(f"{out_dir}/old_model.pt")
            Path(f"{out_dir}/opt.pt").rename(f"{out_dir}/old_opt.pt")
            Path(f"{out_dir}/lrs.pt").rename(f"{out_dir}/old_lrs.pt")
        except FileNotFoundError:
            pass
        torch.save(model.state_dict(), f"{out_dir}/model.pt")
        torch.save(opt.state_dict(), f"{out_dir}/opt.pt")
        torch.save(lrs.state_dict(), f"{out_dir}/lrs.pt")
        toc = time.time()
        print(f"checkpointing took {toc - tic:.3f} s")

    for batch_idx in tqdm(range(num_batches)):
        # if is_ddp:
        #     model.require_backward_grad_sync = (batch_idx + 1) % grad_acc_steps == 0

        global_batch_idx = batch_idx * ddp_world_size + ddp_rank
        begin_idx = global_batch_idx * num_batch_tokens
        end_idx = begin_idx + num_batch_tokens
        batch = data[begin_idx : end_idx + 1]
        batch = batch.to(device)
        inputs = batch[:-1].view(batch_size, ctx_len)
        labels = batch[1:].view(batch_size, ctx_len)

        with model.no_sync():
            logits = model(inputs).logits
            loss = F.cross_entropy(logits.transpose(1, 2), labels)
            loss.backward()

        if (batch_idx + 1) % grad_acc_steps == 0:
            if is_master_process:
                wandb.log(
                    {
                        "epochs": (batch_idx + 1) / num_batches,
                        "lr": lrs.get_last_lr()[0],
                        "loss": loss.item(),
                    }
                )

            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()
            lrs.step()
            opt.zero_grad(set_to_none=True)

        if (batch_idx + 1) % save_every_n_batches == 0:
            if is_master_process:
                checkpoint()

    if is_master_process:
        checkpoint()

    if is_ddp:
        destroy_process_group()
