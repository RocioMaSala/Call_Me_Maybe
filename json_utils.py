

def escape_json_string(text: str) -> str:
    text_converted_barra = text.replace('\\', '\\\\')
    text_converted_quotes = text_converted_barra.replace('"', '\\"')
    return text_converted_quotes
