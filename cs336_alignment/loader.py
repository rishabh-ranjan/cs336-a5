import json
import random

from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

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

    torch.save(tokens_list, out_file)


class MySFTDataset(Dataset):
    def __init__(
        self,
        pt_file,
        ctx_len,
        shuffle,
    ):
        self.ctx_len = ctx_len

        tokens_list = torch.load(pt_file)

        if shuffle:
            random.shuffle(tokens_list)

        self.tokens = []
        for tokens in tokens_list:
            self.tokens.extend(tokens)
        self.tokens = torch.tensor(self.tokens)

    def __len__(self):
        return (len(self.tokens) - 1) // self.ctx_len

    def __getitem__(self, i):
        return self.tokens[i * self.ctx_len : (i + 1) * self.ctx_len + 1]


class SFTDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        dataset_path,
        seq_length,
        shuffle,
    ):

        with open(dataset_path, "r") as f:
            lines = f.readlines()

        docs = []
        for line in lines:
            record = json.loads(line)
            doc = templates.alpaca_chat.format(**record)
            docs.append(doc)

        if shuffle:
            random.shuffle(docs)

        text = ""
        for doc in docs:
            text += "<|begin_of_text|>"
            text += doc
            text += "<|end_of_text|>"

        self.input_ids = tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
        self.seq_length = seq_length

    def __len__(self):
        return (len(self.input_ids) - 1) // self.seq_length

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError
        l = self.seq_length
        input_ids = self.input_ids[i * l : (i + 1) * l]
        labels = self.input_ids[i * l + 1 : (i + 1) * l + 1]
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
