import json
import random

import torch
from torch.utils.data import Dataset, DataLoader

from . import templates


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
