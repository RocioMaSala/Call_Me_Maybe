"""Vocabulary loading and conversion between token ids and text."""
import json
from llm_sdk import Small_LLM_Model


def load_vocab(model: Small_LLM_Model) -> dict[str, int]:
    """Load the vocabulary file of the model.

    Args:
        model: Model whose vocabulary file is read.

    Returns:
        A dictionary that maps each raw token string to its id.

    Raises:
        FileNotFoundError: If the vocabulary file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)
    return vocab


def build_id_to_token(vocab: dict[str, int]) -> dict[int, str]:
    """Invert the vocabulary.

    Args:
        vocab: Dictionary that maps raw token strings to ids.

    Returns:
        A dictionary that maps each id to its raw token string.
    """
    id_to_token = {value: key for key, value in vocab.items()}
    return id_to_token


def token_to_text(id_to_token: dict[int, str], token_id: int) -> str:
    """Return the real text of a token.

    The tokenizer writes a space as 'Ġ' and a line break as 'Ċ'. Both
    markers are replaced by the characters they stand for. Other special
    markers are not converted.

    Args:
        id_to_token: Dictionary that maps ids to raw token strings.
        token_id: Id of the token.

    Returns:
        The text of the token, with spaces and line breaks restored.

    Raises:
        KeyError: If the id is not in the dictionary.
    """
    texto_crudo = id_to_token[token_id]
    texto_limpio = texto_crudo.replace('Ġ', ' ')
    texto_limpio = texto_limpio.replace('Ċ', '\n')
    return texto_limpio


def build_unknown_ids(
        id_to_token: dict[int, str], logits_size: int
        ) -> list[int]:
    """List the positions of the logits that have no text.

    The model returns more logits than the vocabulary file has entries.
    The extra positions are special tokens with no text, and they must
    be blocked so that they are never selected.

    Args:
        id_to_token: Dictionary that maps ids to raw token strings.
        logits_size: Number of logits returned by the model.

    Returns:
        The ids from 0 to logits_size - 1 that are not in id_to_token.
    """
    unknown_ids = [i for i in range(logits_size) if i not in id_to_token]
    return unknown_ids


def build_id_to_text(id_to_token: dict[int, str]) -> dict[int, str]:
    """Convert every token to its real text, once.

    The result is computed a single time so the generation loops do not
    clean the same token again at every step.

    Args:
        id_to_token: Dictionary that maps ids to raw token strings.

    Returns:
        A dictionary that maps each id to the text of its token, with
        spaces and line breaks restored.
    """
    return {id: token_to_text(id_to_token, id) for id in id_to_token}
