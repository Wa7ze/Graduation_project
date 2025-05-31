import csv
import gzip
import itertools
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


class CategoryDataset(Dataset):
    def __init__(self,
                 root_dir="../data/images/",
                 data_file='train_no_dup_with_category_3more_name.json',
                 data_dir="data",
                 transform=None,
                 use_mean_img=True,
                 neg_samples=True):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.transform = transform
        self.use_mean_img = use_mean_img
        self.neg_samples = neg_samples

        full_path = os.path.join(data_dir, data_file)
        print("Looking for:", full_path)

        with open(full_path) as f:
            json_data = json.load(f)

        print("Total raw entries in", data_file, ":", len(json_data))

        self.data = [(k, v) for k, v in json_data.items()]
        print("Sample entry:", self.data[0])

        self.vocabulary, self.word_to_idx = [], {}
        self.word_to_idx['UNK'] = len(self.word_to_idx)
        self.vocabulary.append('UNK')

        with open(os.path.join(self.data_dir, 'final_word_dict.txt')) as f:
            for line in f:
                name = line.strip().split()[0]
                if name not in self.word_to_idx:
                    self.word_to_idx[name] = len(self.word_to_idx)
                    self.vocabulary.append(name)

        print("✔ Dataset loaded with", len(self.data), "samples")

    def __getitem__(self, index):
        attempt = 0
        while attempt < 20:
            try:
                set_id, parts = self.data[index]
                corrupt = self.neg_samples and random.random() < 0.5

                imgs = []
                labels = []
                names = []
                was_corrupted = False

                for part in ['upper', 'bottom', 'shoe', 'bag', 'accessory']:
                    original_part = parts.get(part)
                    corrupt_this = corrupt and original_part is not None and random.random() < 0.6

                    if corrupt_this:
                        for _ in range(10):
                            alt = random.choice(self.data)
                            if alt[0] != set_id and part in alt[1]:
                                alt_item = alt[1][part]
                                alt_path = os.path.join(self.root_dir, str(alt[0]), f"{alt_item['index']}.jpg")
                                if os.path.exists(alt_path):
                                    img_path = alt_path
                                    name = alt_item['name']
                                    labels.append(f"{alt[0]}_{alt_item['index']}")
                                    was_corrupted = True
                                    break
                        else:
                            corrupt_this = False  # fallback to original

                    if not corrupt_this and original_part:
                        img_path = os.path.join(self.root_dir, str(set_id), f"{original_part['index']}.jpg")
                        name = original_part['name']
                        labels.append(f"{set_id}_{original_part['index']}")
                    elif not original_part and self.use_mean_img:
                        img_path = os.path.join(self.data_dir, f"{part}.png")
                        name = ''
                        labels.append(f"{part}_mean")
                    else:
                        continue

                    if not os.path.exists(img_path):
                        raise FileNotFoundError(f"Missing image: {img_path}")

                    img = Image.open(img_path).convert('RGB')
                    img = self.transform(img)
                    imgs.append(img)
                    names.append(torch.LongTensor(self.str_to_idx(name)))

                if not imgs:
                    raise RuntimeError("Empty image set.")

                input_images = torch.stack(imgs)
                offsets = torch.LongTensor(list(itertools.accumulate([0] + [len(n) for n in names[:-1]])))
                is_compat = not was_corrupted

                print(f"[Sample {index}] is_compat = {is_compat}")

                return input_images, names, offsets, set_id, labels, is_compat

            except Exception as e:
                print(f"[Skipping corrupt or missing sample {index}] Reason: {e}")
                index = (index + 1) % len(self.data)
                attempt += 1

        raise RuntimeError("Too many failed attempts to load data.")

    def __len__(self):
        return len(self.data)

    def str_to_idx(self, name):
        return [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx['UNK']
                for w in name.split()]

def collate_fn(data):
    data.sort(key=lambda x: x[0].shape[0], reverse=True)
    images, names, offsets, set_ids, labels, is_compat = zip(*data)
    lengths = [i.shape[0] for i in images]
    is_compat = torch.LongTensor(is_compat)
    names = sum(names, [])
    offsets = list(offsets)
    images = torch.stack(images)
    return (
        lengths,
        images,
        names,
        offsets,
        set_ids,
        labels,
        is_compat
    )
