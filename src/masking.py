import numpy as np
from .tokenizer_utils import token_to_text, build_id_to_token
from llm_sdk import Small_LLM_Model
from .parser import FunctionDefinition


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
    unknown_ids: list[int]
) -> tuple[str, list[int]]:
    candidatos_vivos = []
    for function in functions:
        candidatos_vivos.append(function.name)
    texto_generado = ""
    quote_id = vocab['"']
    id_to_token = build_id_to_token(vocab)
    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        indices_ordenados = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
        top_8 = indices_ordenados[:8]
        for i in unknown_ids:
            logits[i] = float('-inf')
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


def mask_number(
    input_ids: list[int],
    vocab: dict[str, int],
    model: Small_LLM_Model,
    id_to_text: dict[int, str],
) -> tuple[str, list[int]]:
    texto_generado = ""
    comma_id = vocab[',']
    key_id = vocab['}']
    digits_ids = [id for texto, id in vocab.items() if texto.isascii() and texto.isdigit()]

    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        mask = list_creation(logits)

        for id in digits_ids:
            mask[id] = logits[id]
        if len(texto_generado) > 0:
            mask[comma_id] = logits[comma_id]
            mask[key_id] = logits[key_id]

        winner_id = int(np.argmax(mask))
        if winner_id == comma_id or winner_id == key_id:
            break
        texto_generado += id_to_text[winner_id]
        input_ids.append(winner_id)

    return (texto_generado, input_ids)


MAX_STRING_TOKENS = 60


def es_cierre(texto: str) -> bool:
    limpio = texto.replace('Ġ', ' ').replace('Ċ', '\n')
    return (
        limpio.startswith('"')
        and '\\' not in limpio
        and '"' not in limpio[1:]
        and not any(c.isalnum() for c in limpio[1:])
    )

def solo_barras_pares(texto: str) -> bool:
    return len(texto) > 0 and set(texto) == {'\\'} and len(texto) % 2 == 0

def mask_string(
    input_ids: list[int],
    vocab: dict[str, int],
    model: Small_LLM_Model,
    unknown_ids: list[int],
) -> tuple[str, list[int]]:
    texto_generado = ""
    barras_id = vocab['\\']
    id_to_token = build_id_to_token(vocab)
    after_backslash = False
    pasos = 0

    while True:
        pasos += 1
        if pasos > MAX_STRING_TOKENS:
            raise ValueError(
                f"El string superó {MAX_STRING_TOKENS} tokens sin cerrarse: "
                f"{texto_generado!r}"
            )

        logits = model.get_logits_from_input_ids(input_ids)
        crudos = list(logits)
        for i in unknown_ids:
            logits[i] = float('-inf')

        for texto_token, id in vocab.items():
            if not after_backslash:
                es_valido = (
                    es_cierre(texto_token)
                    or texto_token == '\\'
                    or solo_barras_pares(texto_token)
                    or ('"' not in texto_token and '\\' not in texto_token)
                )
            else:
                es_valido = texto_token in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't']
            if not es_valido:
                logits[id] = float('-inf')

        winner_id = int(np.argmax(logits))
        if pasos <= 2:
            top = sorted(range(len(crudos)), key=lambda i: crudos[i],
                         reverse=True)[:6]
            print("CRUDOS:", [id_to_token.get(i, '<esp>') for i in top],
                  "->", id_to_token[winner_id])
        if not after_backslash and es_cierre(id_to_token[winner_id]):
            break

        texto_generado += token_to_text(id_to_token, winner_id)
        input_ids.append(winner_id)
        after_backslash = (winner_id == barras_id) and not after_backslash

    return (texto_generado, input_ids)


def write_literal(
    input_ids: list[int],
    model: Small_LLM_Model,
    text: str,
) -> tuple[str, list[int]]:
    """Añade a input_ids un texto fijo ya tokenizado, sin consultar al modelo."""
    input_ids.extend(model.encode(text).tolist()[0])
    return (text, input_ids)

