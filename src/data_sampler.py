import math
import torch
import json
import os.path
from typing import List, Union, Dict, Mapping, Optional, Tuple, TypedDict
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from typing import cast, List, Dict, Union, Dict, Optional, Mapping
import pandas as pd
import ir_datasets
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
# from datasets import load_dataset, Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer, AutoTokenizer
from tqdm import tqdm

@dataclass
class CorpusCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)
        self.passage_max_len = 512

    def __call__(self, features):
        ids = [f[0] for f in features]
        passage = [f[1] for f in features]
        ids = torch.tensor(ids, dtype=torch.int)
        d_collated = self.tokenizer(    
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"ids": ids, "passage": d_collated}

@dataclass
class CorpusCollatorNV(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    def __init__(self, tokenizer, p_instruct, padding_side, is_mask_instruction, max_length=512, always_add_eos=True):
        super().__init__(tokenizer=tokenizer)
        self.p_instruct = p_instruct
        self.padding_side = padding_side
        self.is_mask_instruction = is_mask_instruction
        self.max_length = max_length
        self.always_add_eos = True
        if self.padding_side == "right" and self.is_mask_instruction == True and len(p_instruct) > 0:
            self.instruction_lens = len(self.tokenizer.tokenize(p_instruct))
        else:
            self.instruction_lens = 0
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = self.padding_side

    def __call__(self, features):
        ids = [f[0] for f in features]
        passage = [f[1] for f in features]
        ids = torch.tensor(ids, dtype=torch.int)

        if self.always_add_eos:
            passage = [self.p_instruct + input_example + self.tokenizer.eos_token for input_example in passage]
        
        batch_dict = self.tokenizer(
            passage,
            max_length=self.max_length,
            padding=True,
            return_token_type_ids=False,
            return_tensors="pt",
            truncation=True)
        
        attention_mask = batch_dict['attention_mask'].clone() if 'attention_mask' in batch_dict else None
        if (attention_mask is not None and
            self.padding_side == "right" and
            self.is_mask_instruction == True and
            self.instruction_lens > 0):
            # Mask out the instruction tokens for mean-pooling
            attention_mask[:, :self.instruction_lens] = 0
        
        features = {
            'input_ids': batch_dict.get('input_ids'),
            'attention_mask': batch_dict['attention_mask'],
            'pool_mask': attention_mask,
        }
        return {"ids" : ids, "passage": features}

class CorpusDatasetForEmbedding(Dataset):
    def __init__(
        self,
        docstore,
        n_docs
    ):
        self.docstore = docstore
        self.n_docs = n_docs

    def __len__(self):
        return self.n_docs

    def __getitem__(self, item):
        item += 1
        text = self.docstore.get(str(item)).text
        return item, text

class QueryDatasetForEmbedding(Dataset):
    def __init__(
        self,
        query_dict,
    ):
        self.queries = []
        for k in query_dict:
            self.queries.append([k, query_dict[k]['q']])
            
        # self.docstore = docstore
        # self.n_docs = n_docs

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, item):
        id, text = self.queries[item]
        return int(id), text
