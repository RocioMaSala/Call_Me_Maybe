from .parser import FunctionDefinition, ParameterSchema

def add_json_instruction(prompt: str) -> str:
    instruccion = (
        "\n\nRespond with a single JSON object, with no text before or after it.\n"
        "Example:\n"
        "User request: Replace all spaces in 'a b c' with '-'\n"
        'Response: {"prompt": "Replace all spaces in \'a b c\' with \'-\'", '
        '"name": "fn_substitute_string_with_regex", "parameters": '
        '{"source_string": "a b c", "regex": "\\\\s", "replacement": "-"}}\n'
    )
    return prompt + instruccion

def format_parameters(parameters: dict[str, ParameterSchema]) -> str:
    prompt_parameters = [f"{name}: {value.type}" for name, value in parameters.items()]
    return ", ".join(prompt_parameters)

def format_function_compact(func: FunctionDefinition) -> str:
    result = f"- {func.name}({format_parameters(func.parameters)}): {func.description}"
    return result

def build_functions_section(definitions: list[FunctionDefinition]) -> str:
    prompt_functions = f"Available functions:\n"
    for function in definitions:
        prompt_functions += f"{format_function_compact(function)}\n"
    return prompt_functions

def build_prompt_compact(user_prompt: str, definitions: list[FunctionDefinition]) -> str:
    prompt_complete_compact = f"{build_functions_section(definitions)}"
    prompt_complete_compact += f"\nUser request: {user_prompt}"
    return prompt_complete_compact

# Analizar cuando tenga la máscara, qué tipo de prompt es mejor que le pase a la máquina.
def format_function_explicit(func: FunctionDefinition) -> str:
    result = f"Function name: {func.name} \nParameters: {format_parameters(func.parameters)} \nDescription: {func.description}"
    return result

def build_functions_section_explicit(definitions: list[FunctionDefinition]) -> str:
    prompt_functions = f"Available functions:\n"
    for function in definitions:
        prompt_functions += f"{format_function_explicit(function)}\n"
    return prompt_functions

def build_prompt_explicit(user_prompt: str, definitions: list[FunctionDefinition]) -> str:
    prompt_complete_explicit = f"{build_functions_section_explicit(definitions)}"
    prompt_complete_explicit += f"\nUser request: {user_prompt}"
    return prompt_complete_explicit

def build_prompt_final(user_prompt: str, definitions: list[FunctionDefinition]) -> str:
    return (
        build_functions_section(definitions)
        + "\nRespond with a single JSON object, with no text before or after it.\n"
        "Copy text from the request exactly into string arguments. Never shorten it.\n"
        "Example:\n"
        "User request: Replace all spaces in 'a b c' with '-'\n"
        'Response: {"prompt": "Replace all spaces in \'a b c\' with \'-\'", '
        r'"name": "fn_example_replace", "parameters": {"source_string": "a b c", '
        r'"regex": "\\s", "replacement": "-"}}'
        "\n\n"
        f"User request: {user_prompt}\n"
        "Response: "
    )

