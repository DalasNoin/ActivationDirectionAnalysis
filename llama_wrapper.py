import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM


class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None


class LlamaWrapper:
    def __init__(
        self,
        token,
        system_prompt,
        model_name:str,
        size="7b",
        use_chat=True,
    ):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.use_chat = use_chat
        if model_name:
            self.model_name_path = model_name
        elif use_chat:
            self.model_name_path = f"meta-llama/Llama-2-{size}-chat-hf"
        else:
            self.model_name_path = f"/data/jeffrey_ladish_stream/lora_models/7b_merge_eos"
        self.name = self.model_name_path.split("/")[-1]
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf", token=token
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name_path, token=token
            )
            .half()
            .to(self.device)
        )
        self.layer_count = len(self.model.model.layers)
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer)

    def get_logits(self, tokens):
        with t.no_grad():
            logits = self.model(tokens).logits
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()