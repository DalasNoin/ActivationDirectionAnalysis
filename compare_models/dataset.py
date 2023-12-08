"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Usage:
python generate_vectors.py --layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --use_base_model --model_size 7b --data_path datasets/test.json --dataset_name test
"""

import json
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv
from typing import List
from utils.tokenize_llama import tokenize_llama

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
SAVE_VECTORS_PATH = "vectors"


class BaseDataset(Dataset):
    def __init__(self, 
                 data_path:str, 
                 system_prompt:str, 
                 token:str, 
                 model_name_path: str, 
                 use_chat: bool):
        
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_path, token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat

    def prompt_to_tokens(self, instruction, model_output=""):

        tokens = tokenize_llama(
            self.tokenizer,
            self.system_prompt,
            [(instruction, model_output)],
            no_final_eos=True,
            chat_model=self.use_chat,
        )
        return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        raise NotImplementedError()

class SimpleDataset(BaseDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        q_text = item["question"]
        tokens = self.prompt_to_tokens(q_text)
        return tokens
    
class ComparisonDataset(Dataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item["question"]
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens, n_tokens