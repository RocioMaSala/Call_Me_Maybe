import tokenizer_utils
import numpy as np
from llm_sdk import Small_LLM_Model

def list_creation (logits: list[float]) -> list[float]:
    comparison_list = [float('-inf')] * len(logits)
    return comparison_list

def mask_boolean (input_ids: list[int], vocab: dict[str, int], model: Small_LLM_Model) -> tuple[str, list[int]]:
    logits = model.get_logits_from_input_ids(input_ids)
    base_mask = list_creation(logits)
    true_id = vocab['true']
    false_id = vocab['false']
    base_mask[true_id] = logits[true_id]
    base_mask[false_id] = logits[false_id]
    winner_id = np.argmax(base_mask)
    if winner_id == true_id:
        winner_str = "true"
    else:
        winner_str = "false"
    input_ids.append(winner_id)
    return (winner_str, input_ids)
