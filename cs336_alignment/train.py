import pkg_resources

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .loader import MySFTDataset


def main(
    hf_model="meta-llama/Meta-Llama-3-8B",
    ctx_len=512,
    batch_size=2,
    grad_acc_steps=16,
    lr=2e-5,
):
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    net = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print("init net")
    net = net.to(device)
    print("move net to device")

    train_file = pkg_resources.resource_filename("cs336_alignment", "data/sft/train.pt")
    dataset = MySFTDataset(train_file, ctx_len, shuffle=True)
    print("init dataset")
    loader = DataLoader(dataset, batch_size, shuffle=True)
    print("init loader")

    opt = optim.AdamW(net.parameters(), lr=lr)
    print("init opt")
    # lrs = optim.lr_scheduler.CosineAnnealingLR(opt)

    for batch_idx, batch in enumerate(loader):
        print(batch_idx)
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        logit = net(x).logits
        loss = F.cross_entropy(logit, y)

        print(f"\r{batch_idx: 5}: {loss.item():.3f}    ", end="")
        loss.backward()
        opt.step()
        opt.zero_grad()

        # if (batch_idx + 1) % grad_acc_steps == 0:
        #     opt.step()
        #     opt.zero_grad()
