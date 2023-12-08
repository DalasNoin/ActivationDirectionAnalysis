"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Usage:
python generate_vectors.py --layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --use_base_model --model_size 7b --data_path datasets/test.json --dataset_name test
"""

import torch
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
from compare_models.dataset import SimpleDataset

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
SAVE_VECTORS_PATH = "vectors"


def generate_vectors(
    data_path: str, 
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

    activations = dict([(layer, []) for layer in layers])

    dataset = SimpleDataset(
        data_path=data_path,
        system_prompt=SYSTEM_PROMPT,
        token=HUGGINGFACE_TOKEN,
        model_name_path=model.model_name_path,
        use_chat=model.use_chat,
    )
    dataset_logits = []
    for tokens in tqdm(dataset, desc="Processing prompts"):
        tokens = tokens.to(model.device)
        model.reset_all()
        logits = model.get_logits(tokens).detach().cpu()
        dataset_logits.append(logits[0, -1, :])
        for layer in layers:
            activations = model.get_last_activations(layer)
            activations = activations[0, -1, :].detach().cpu()
            activations[layer].append(activations)

    # take mean of logits, then softmax
    dataset_logits = torch.stack(dataset_logits)
    dataset_logits = dataset_logits.mean(dim=0).softmax(dim=-1)

    
    os.makedirs(os.path.join(SAVE_VECTORS_PATH, model.name, dataset_name), exist_ok=True)
    vectors = {}
    for layer in layers:
        all_layers = torch.stack(activations[layer])
        vec = (all_layers).mean(dim=0)
        vectors[layer] = vec
        # t.save(
        #     vec,
        #     os.path.join(
        #         SAVE_VECTORS_PATH,
        #         model.name,
        #         dataset_name,
        #         f"vec_layer_{layer}.pt",
        #     ),
        # )
    return vectors, dataset_logits