"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Usage:
python generate_vectors.py --layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --use_base_model --model_size 7b --data_path datasets/test.json --dataset_name test
"""

import json
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from llama_wrapper import LlamaWrapper
import argparse
from typing import List
from utils.tokenize_llama import tokenize_llama
from utils.helpers import make_tensor_save_suffix

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
SAVE_VECTORS_PATH = "vectors"


class ComparisonDataset(Dataset):
    def __init__(self, data_path, system_prompt, token, model_name_path, use_chat):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_path, token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat

    def prompt_to_tokens(self, instruction, model_output):

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
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item["question"]
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens, n_tokens


def generate_save_vectors(
    data_path: str, 
    # dataset_name: str,
    model_name: str,
    layers: List[int] = None,
    base_model: bool = False,
):
    """
    layers: list of layers to generate vectors for
    data_path: path to dataset
    dataset_name: name of dataset
    model_name: name of model to use
    """
    if not os.path.exists(SAVE_VECTORS_PATH):
        os.makedirs(SAVE_VECTORS_PATH)

    dataset_name = data_path.split("/")[-1].split(".")[0]

    model = LlamaWrapper(
        HUGGINGFACE_TOKEN, SYSTEM_PROMPT, model_name=model_name, use_chat=not base_model
    )
    model.reset_all()
    if layers is None:
        layers = list(range(model.layer_count))

    pos_activations = dict([(layer, []) for layer in layers])
    neg_activations = dict([(layer, []) for layer in layers])

    dataset = ComparisonDataset(
        data_path,
        SYSTEM_PROMPT,
        HUGGINGFACE_TOKEN,
        model.model_name_path,
        model.use_chat,
    )

    for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)
        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer)
            p_activations = p_activations[0, -1, :].detach().cpu()
            pos_activations[layer].append(p_activations)
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, -1, :].detach().cpu()
            neg_activations[layer].append(n_activations)
    
    os.makedirs(os.path.join(SAVE_VECTORS_PATH, model.name, dataset_name), exist_ok=True)
    for layer in layers:
        all_pos_layer = t.stack(pos_activations[layer])
        all_neg_layer = t.stack(neg_activations[layer])
        vec = (all_pos_layer - all_neg_layer).mean(dim=0)
        t.save(
            vec,
            os.path.join(
                SAVE_VECTORS_PATH,
                model.name,
                dataset_name,
                f"vec_layer_{layer}.pt",
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--data_path", type=str, default="datasets/test.json")
    parser.add_argument("--use_base_model", action="store_true")
    # parser.add_argument("--dataset_name", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Huggingface model name or path")

    args = parser.parse_args()
    generate_save_vectors(
        layers=args.layers, 
        data_path=args.data_path, 
        # dataset_name=args.dataset_name, 
        model_name=args.model_name,
        base_model=args.use_base_model,
    )
