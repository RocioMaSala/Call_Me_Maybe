from .parser import FunctionDefinition, ParameterSchema

def add_json_instruction(prompt: str) -> str:
    instruccion = (
        "\n\nRespond with a single JSON object, with no text before or after it.\n"
        "Example:\n"
        'User request: What is the sum of 10 and 20?\n'
        'Response: {"prompt": "What is the sum of 10 and 20?", "name": "fn_example_function", "parameters": {"param1": 10, "param2": 20}}\n'
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

