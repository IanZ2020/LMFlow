import random
import numpy as np
from itertools import chain
import torch
import json
from datasets import load_dataset
from lmflow.datasets.dataset import Dataset
from lmflow.args import DatasetArguments

def get_c4(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0 )

def get_wikitext(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'wikitext', 'wikitext-103-raw-v1', split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0 )

def get_wikitext_cat(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'wikitext', 'wikitext-103-raw-v1', split='train'
    )
    l = 50
    tokenized_samples, history = [], []
    for j in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - l)
            tokenized_sample = tokenizer(' '.join(traindata[i:i+l]['text']), return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.extend(list(range(i,i+l)))
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
        print(f'{j}/{n_samples}')
    return torch.cat(tokenized_samples, dim=0 )

def get_redpajama(tokenizer, n_samples, seq_len):
    dataset_path = 'data/redpajama_mini_formatted'
    block_size = seq_len
    data_args = DatasetArguments(dataset_path=dataset_path, block_size=block_size, max_train_samples=n_samples)
    dataset = Dataset(data_args)
    traindata = dataset.get_backend_dataset()

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        group_text = []
        group_len = 0
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')

            if i not in history:
                sample_len = tokenized_sample.input_ids.shape[1]
                group_len += sample_len
                history.append(i)
                group_text.append(tokenized_sample.input_ids.squeeze())
            
            if group_len >= seq_len:
                break
        group_text = torch.cat(group_text,dim=0)
        i = random.randint(0, group_len - seq_len)
        tokenized_samples.append(group_text[i:i+seq_len].unsqueeze(dim = 0))
    return torch.cat(tokenized_samples, dim=0)

def get_bookcorpus_cat(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train'
    )
    l = 50
    tokenized_samples, history = [], []
    for j in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - l)
            tokenized_sample = tokenizer(' '.join(traindata[i:i+l]['text']), return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.extend(list(range(i,i+l)))
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
        print(f'{j}/{n_samples}')
    return torch.cat(tokenized_samples, dim=0 )


def get_examples(dataset, tokenizer, n_samples, seq_len = 128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus_cat':
        return get_bookcorpus_cat(tokenizer, n_samples, seq_len)
    elif dataset == 'redpajama':
        return get_redpajama(tokenizer, n_samples, seq_len)
    elif dataset == 'wikitext':
        return get_wikitext(tokenizer, n_samples, seq_len)
    elif dataset == 'wikitext_cat':
        return get_wikitext(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError
