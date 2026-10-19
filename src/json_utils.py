"""Helpers to build valid JSON text."""


def escape_json_string(text: str) -> str:
    """Escape a text so it can be embedded inside a JSON string.

    Backslashes are escaped first and double quotes afterwards, so the
    backslashes added in front of the quotes are not escaped twice.
    Other control characters (line breaks, tabs) are not handled.

    Args:
        text: Raw text, as read from the input file.

    Returns:
        The text with every backslash and double quote escaped.
    """
    text_converted_barra = text.replace('\\', '\\\\')
    text_converted_quotes = text_converted_barra.replace('"', '\\"')
    return text_converted_quotes
