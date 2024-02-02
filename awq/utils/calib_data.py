import torch
import logging
from typing import List, Union
from datasets import load_dataset

def get_calib_dataset(data: Union[str, List[str], List[List[int]]] = "pileval",
                      tokenizer=None, n_samples=512*5, block_size=512,
                      split="train", text_column="text"):
    print('!XXX FIXED')
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        else:
            dataset = load_dataset(data, split=split)
        
        dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset =  data
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words.")
    else:
        raise NotImplementedError(
            "Either pass a string to a huggingface dataset or a list"
            "that is preprocessed with one sample of text per element"
            " or a list of list of int for tokenized words.")
    
    samples = []
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        # Split line_encoded into chunks of at most block_size tokens
        for i in range(0, len(line_encoded), block_size):
            sample_raw = line_encoded[i:i+block_size]
            if len(sample_raw) < block_size * 0.75:
                continue
            sample = torch.tensor([sample_raw])
            samples.append(sample)
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    logging.debug(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]
