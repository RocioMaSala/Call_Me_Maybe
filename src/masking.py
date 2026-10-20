
"""Constrained-decoding functions: one per type of value to generate."""
import numpy as np
from .tokenizer_utils import build_id_to_token
from llm_sdk import Small_LLM_Model
from .parser import FunctionDefinition

MAX_STRING_TOKENS = 60


def list_creation(logits: list[float]) -> list[float]:
    """Create a mask in which every token is blocked.

    Args:
        logits: Logits returned by the model. Only its length is used.

    Returns:
        A list of the same length filled with minus infinity.
    """
    comparison_list = [float("-inf")] * len(logits)
    return comparison_list


def mask_boolean(
    input_ids: list[int], vocab: dict[str, int], model: Small_LLM_Model
) -> tuple[str, list[int]]:
    """Choose between true and false.

    Only the tokens "true" and "false" are left available, and the
    model picks the one with the higher logit. The chosen token is
    appended to input_ids.

    Args:
        input_ids: Token ids of the context. It is modified in place.
        vocab: Maps raw token strings to ids.
        model: Model that returns the logits.

    Returns:
        The chosen value as "true" or "false", and input_ids.
    """
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
    """Generate the name of one of the available functions.

    At each step only the tokens that continue the name of at least one
    candidate are allowed. When the text generated so far is exactly a
    full name, the closing quote is allowed too, and the model decides
    whether to stop there. The closing quote is not added to input_ids.

    Args:
        input_ids: Token ids of the context. It is modified in place.
        vocab: Maps raw token strings to ids.
        model: Model that returns the logits and decodes the ids.
        functions: The available functions.
        unknown_ids: Positions of the logits that have no text.

    Returns:
        The generated function name, and input_ids.
    """
    candidatos_vivos = []
    for function in functions:
        candidatos_vivos.append(function.name)
    ids_generados: list[int] = []
    texto_generado = ""
    quote_id = vocab['"']
    while True:
        logits = model.get_logits_from_input_ids(input_ids)
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
        ids_generados.append(winner_id)
        texto_generado = model.decode(ids_generados)
        candidatos_vivos = [c for c in candidatos_vivos if
                            c.startswith(texto_generado)]
        input_ids.append(winner_id)
    return (texto_generado, input_ids)


def mask_number(
    input_ids: list[int],
    vocab: dict[str, int],
    model: Small_LLM_Model,
) -> tuple[str, list[int]]:
    """Generate a number, one digit at a time.

    Only the ASCII digits are allowed. Once there is at least one
    digit, a comma or a closing brace is allowed too, and choosing one
    of them ends the number. That token is not added to input_ids.

    Only non-negative integers can be generated: the minus sign and the
    decimal point are never allowed. There is no limit on the number of
    digits.

    Args:
        input_ids: Token ids of the context. It is modified in place.
        vocab: Maps raw token strings to ids.
        model: Model that returns the logits and decodes the ids.

    Returns:
        The digits generated, as a string, and input_ids.
    """
    ids_generados: list[int] = []
    comma_id = vocab[',']
    key_id = vocab['}']
    digits_ids = [id for texto, id in vocab.items()
                  if texto.isascii() and texto.isdigit()]

    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        mask = list_creation(logits)

        for id in digits_ids:
            mask[id] = logits[id]
        if len(ids_generados) > 0:
            mask[comma_id] = logits[comma_id]
            mask[key_id] = logits[key_id]

        winner_id = int(np.argmax(mask))
        if winner_id == comma_id or winner_id == key_id:
            break
        ids_generados.append(winner_id)
        input_ids.append(winner_id)
    return (model.decode(ids_generados), input_ids)


def es_cierre(texto: str) -> bool:
    """Tell whether a token closes a string.

    A token closes a string when it starts with a double quote and
    carries nothing else that matters, such as '"', '",' or '"}'.
    Tokens such as '"a', which put text after the quote, do not close it.

    Args:
        texto: Raw token string, as it appears in the vocabulary.

    Returns:
        True if the token closes a string.
    """
    limpio = texto.replace('Ġ', ' ').replace('Ċ', '\n')
    return (
        limpio.startswith('"')
        and '\\' not in limpio
        and '"' not in limpio[1:]
        and not any(c.isalnum() for c in limpio[1:])
    )


def solo_barras_pares(texto: str) -> bool:
    """Tell whether a token is made only of an even number of backslashes.

    Two backslashes in a row are a valid JSON escape for one backslash.

    Args:
        texto: Raw token string.

    Returns:
        True if the token has only backslashes and their count is even.
    """
    return len(texto) > 0 and set(texto) == {'\\'} and len(texto) % 2 == 0


def mask_string(
    input_ids: list[int],
    vocab: dict[str, int],
    model: Small_LLM_Model,
    unknown_ids: list[int],
) -> tuple[str, list[int]]:
    """Generate the content of a JSON string.

    Allowed tokens: tokens that close the string, a single backslash,
    tokens made of an even number of backslashes, and any token that
    has neither a quote nor a backslash. Right after a single
    backslash, only the escapes '" \\ / b f n r t' are allowed.

    The token that closes the string is not added to input_ids, so the
    caller has to write the closing quote. Tokens that contain a line
    break are not blocked, and a line break inside a string makes the
    JSON invalid. There is no limit on the length of the string.

    Args:
        input_ids: Token ids of the context. It is modified in place.
        vocab: Maps raw token strings to ids.
        model: Model that returns the logits.
        unknown_ids: Positions of the logits that have no text.

    Returns:
        The text generated, without the closing quote, and input_ids.
    """
    ids_generados: list[int] = []
    barras_id = vocab['\\']
    id_to_token = build_id_to_token(vocab)
    after_backslash = False

    while True:
        logits = model.get_logits_from_input_ids(input_ids)
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
                es_valido = texto_token in ['"',
                                            '\\', '/', 'b', 'f',
                                            'n', 'r', 't']
            if not es_valido:
                logits[id] = float('-inf')

        winner_id = int(np.argmax(logits))
        if not after_backslash and es_cierre(id_to_token[winner_id]):
            break

        ids_generados.append(winner_id)
        input_ids.append(winner_id)
        after_backslash = (winner_id == barras_id) and not after_backslash

    return (model.decode(ids_generados), input_ids)


def write_literal(
    input_ids: list[int],
    model: Small_LLM_Model,
    text: str,
) -> tuple[str, list[int]]:
    """Add a fixed text to the context without querying the model.

    The text is tokenized and its ids are appended to input_ids, so the
    model sees it as context for what comes next.

    Args:
        input_ids: Token ids of the context. It is modified in place.
        model: Model whose tokenizer is used.
        text: Fixed text to add.

    Returns:
        The same text, and input_ids.
    """
    input_ids.extend(model.encode(text).tolist()[0])
    return (text, input_ids)
