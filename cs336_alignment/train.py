import json
import pkg_resources
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn, optim
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


class SFTDataset(Dataset):
    def __init__(
        self,
        data_file,
        ctx_len,
    ):
        self.ctx_len = ctx_len
        self.data = torch.load(data_file)

    def __len__(self):
        return (self.data.size(0) - 1) // self.ctx_len

    def __getitem__(self, i):
        return self.data[i * self.ctx_len : (i + 1) * self.ctx_len + 1]


def main(
    *,
    hf_model="meta-llama/Meta-Llama-3-8B",
    ctx_len=512,
    batch_size=2,
    grad_acc_steps=16,
    lr=2e-5,
    weight_decay=0.01,
    wandb_project="sft",
    seed=42,
    grad_clip=1.0,
):
    if wandb_project:
        wandb.init(project=wandb_project, config=locals())

    torch.manual_seed(seed)

    device = "cuda"

    data_file = pkg_resources.resource_filename("cs336_alignment", f"data/sft/train.pt")
    data = torch.load(data_file).to(device)
    num_batches = (data.size(0) - 1) // ctx_len
    num_steps = num_batches // grad_acc_steps
    num_batch_tokens = batch_size * ctx_len

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device)

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

    for batch_idx in tqdm(range(num_batches)):
        begin_idx = batch_idx * num_batch_tokens
        end_idx = (batch_idx + 1) * num_batch_tokens
        inputs = data[begin_idx:end_idx].view(batch_size, ctx_len)
        labels = data[begin_idx + 1 : end_idx + 1].view(batch_size, ctx_len)

        logits = model(inputs).logits
        loss = F.cross_entropy(logits.transpose(1, 2), labels)
        loss.backward()

        if (batch_idx + 1) % grad_acc_steps == 0:
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
