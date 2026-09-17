import json
from llm_sdk import Small_LLM_Model

def load_vocab(model: Small_LLM_Model) -> dict[str, int]:
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab

def build_id_to_token(vocab: dict[str, int]) -> dict[int, str]:
    id_to_token = {value: key for key, value in vocab.items()}
    return id_to_token
