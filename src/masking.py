from tokenizer_utils import token_to_text, build_id_to_token
import numpy as np
from llm_sdk import Small_LLM_Model
from parser import FunctionDefinition


def list_creation(logits: list[float]) -> list[float]:
    comparison_list = [float("-inf")] * len(logits)
    return comparison_list


def mask_boolean(
    input_ids: list[int], vocab: dict[str, int], model: Small_LLM_Model
) -> tuple[str, list[int]]:
    logits = model.get_logits_from_input_ids(input_ids)
    base_mask = list_creation(logits)
    true_id = vocab["true"]
    false_id = vocab["false"]
    base_mask[true_id] = logits[true_id]
    base_mask[false_id] = logits[false_id]
    winner_id = int(np.argmax(base_mask))
    if winner_id == true_id:
        winner_str = "true"
    else:
        winner_str = "false"
    input_ids.append(winner_id)
    return (winner_str, input_ids)


def mask_name(
    input_ids: list[int],
    vocab: dict[str, int],
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
) -> tuple[str, list[int]]:
    candidatos_vivos = []
    for function in functions:
        candidatos_vivos.append(function.name)
    texto_generado = ""
    quote_id = vocab['"']
    id_to_token = build_id_to_token(vocab)
    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        for texto_token, id in vocab.items():
            es_valido = False
            for candidato in candidatos_vivos:
                resto_esperando = candidato[len(texto_generado):]
                if resto_esperando.startswith(texto_token):
                    es_valido = True
                    break
            if texto_token == '"' and texto_generado in candidatos_vivos:
                es_valido = True
            if not es_valido:
                logits[id] = float('-inf')
        winner_id = int(np.argmax(logits))
        if winner_id == quote_id and texto_generado in candidatos_vivos:
            break
        texto_generado += token_to_text(id_to_token, winner_id)
        candidatos_vivos = [c for c in candidatos_vivos if c.startswith(texto_generado)]
        input_ids.append(winner_id)
    return (texto_generado, input_ids)


def mask_number(input_ids: list[int], vocab: dict[str, int], model: Small_LLM_Model) -> tuple[str, list[int]]:
    texto_generado = ""
    comma_id = vocab[',']
    key_id = vocab['}']
    id_to_token = build_id_to_token(vocab)
    digits_ids = [id for texto, id in vocab.items() if texto.isdigit()]
    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        for texto_token, id in vocab.items():
            es_valido = False
            if id in digits_ids:
                es_valido = True
            if len(texto_generado) > 0 and (id == comma_id or id == key_id):
                es_valido = True
            if not es_valido:
                logits[id] = float('-inf')
        winner_id = int(np.argmax(logits))
        if (winner_id == key_id or winner_id == comma_id) and len(texto_generado) > 0:
            break
        texto_generado += token_to_text(id_to_token, winner_id)
        input_ids.append(winner_id)
    return (texto_generado, input_ids)


def mask_string(input_ids: list[int], vocab: dict[str, int], model: Small_LLM_Model) -> tuple[str, list[int]]:
    texto_generado = ""
    comillas_id = vocab['"']
    barras_id = vocab['\\']
    id_to_token = build_id_to_token(vocab)
    after_backslash = False
    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        for texto_token, id in vocab.items():
            if not after_backslash:
                es_valido = (texto_token == '"') or (texto_token == '\\') or ('"' not in texto_token and '\\' not in texto_token)
                if not es_valido:
                    logits[id] = float('-inf')
            else:
                caracteres_escape_validos = ['"', '\\', '/', 'b', 'f', 'n', 'r', 't']
                es_valido = texto_token in caracteres_escape_validos
                if not es_valido:
                    logits[id] = float('-inf')
        winner_id = int(np.argmax(logits))
        if (winner_id == comillas_id and not after_backslash):
            break
        texto_generado += token_to_text(id_to_token, winner_id)
        input_ids.append(winner_id)
        if winner_id == barras_id:
            after_backslash = True
        else:
            after_backslash = False
    return(texto_generado, input_ids)


def mask_literal(input_ids: list[int], vocab: dict[str, int], model: Small_LLM_Model, text: str) -> tuple[str, list[int]]:

