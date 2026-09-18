import tokenizer_utils
import numpy as np
from llm_sdk import Small_LLM_Model
from parser import FunctionDefinition

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
    winner_id = int(np.argmax(base_mask))
    if winner_id == true_id:
        winner_str = "true"
    else:
        winner_str = "false"
    input_ids.append(winner_id)
    return (winner_str, input_ids)


def mask_name (input_ids: list[int], vocab: dict[str, int], model: Small_LLM_Model, functions: list[FunctionDefinition]) -> tuple[str, list[int]]:
    candidatos_vivos = []
    for function in functions:
        candidatos_vivos.append(function.name)
    texto_generado = ""
    dot_id = vocab['"']
    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        base_mask = list_creation(logits)
        for candidato in candidatos_vivos:
            resto_esperado = candidato[len(texto_generado):]
    


    winner_id = int(np.argmax(base_mask))
    if winner_id == quote_id and texto_generado in candidatos_vivos:
        break
    input_ids.append(winner_id)
    return (winner_str, input_ids)
